"""Four-stage GPT-OSS inference using the actual Hugging Face decoder layers.

One thread owns each GPU's nine layers. Bounded queues overlap microbatches while
preserving input order and capturing the same pre-MLP tensors as the serial model.
"""
from queue import Empty, Full, Queue
import threading

import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


def pipeline_batches(model, dataloader, layers, start_batch, validation, save_validation):
    context = threading.local()
    stop = threading.Event()
    queues = [Queue(maxsize=2) for _ in range(5)]
    handles = []

    def capture(index):
        def hook(module, args, output):
            context.activations[index] = output.detach().to('cpu', torch.float16).reshape(-1, output.shape[-1])
        return hook

    for i in layers:
        handles.append(model.model.layers[i].post_attention_layernorm.register_forward_hook(capture(i)))

    def stage(index, item):
        with torch.inference_mode(), torch.cuda.device(index):
            context.activations = item['activations']
            if index == 0:
                ids = item['batch']['input_ids'].to('cuda:0')
                attention = item['batch']['attention_mask'].to('cuda:0')
                hidden = model.model.embed_tokens(ids)
                positions = torch.arange(hidden.shape[1], device=hidden.device)
                position_ids = positions.unsqueeze(0)
                kwargs = dict(config=model.config, inputs_embeds=hidden,
                              attention_mask=attention, cache_position=positions, past_key_values=None)
                item.update(hidden=hidden, positions=positions, position_ids=position_ids,
                            embeddings=model.model.rotary_emb(hidden, position_ids),
                            masks={'full_attention':create_causal_mask(**kwargs),
                                   'sliding_attention':create_sliding_window_causal_mask(**kwargs)})
            hidden = item['hidden']
            # Accelerate's existing layer hooks move the arguments to this GPU,
            # exactly as in GptOssModel.forward. There is one owner per layer.
            for layer in model.model.layers[index*9:(index+1)*9]:
                hidden = layer(hidden, attention_mask=item['masks'][layer.attention_type],
                               position_embeddings=item['embeddings'], position_ids=item['position_ids'],
                               past_key_values=None, use_cache=False, cache_position=item['positions'])
            item['hidden'] = hidden
            if index == 3:
                # All needed activations precede the final normalization/head.
                for key in ['hidden','positions','position_ids','embeddings','masks']:
                    del item[key]
            torch.cuda.synchronize(index)
            return item

    def put(queue, value):
        while not stop.is_set():
            try:
                queue.put(value, timeout=.2)
                return
            except Full:
                pass

    def fail(error):
        # The main thread always drains this queue, including on worker failure.
        put(queues[-1], error)
        stop.set()

    def producer():
        try:
            for bi, batch in enumerate(dataloader):
                if bi >= start_batch:
                    put(queues[0], {'batch_ix':bi, 'batch':batch, 'activations':{}})
                if stop.is_set():
                    return
            put(queues[0], None)
        except BaseException as error:
            fail(error)

    def worker(index):
        try:
            while not stop.is_set():
                try:
                    item = queues[index].get(timeout=.2)
                except Empty:
                    continue
                if item is None:
                    put(queues[index+1], None)
                    return
                put(queues[index+1], stage(index, item))
        except BaseException as error:
            fail(error)

    threads = []
    try:
        # Compare the entire selected hidden-state tensors before using threads.
        sample, reference = validation
        item = {'batch':{'input_ids':sample.input_ids.cpu(), 'attention_mask':sample.attention_mask.cpu()},
                'activations':{}}
        for i in range(4):
            item = stage(i, item)
        differences = {i:float((reference[i].float()-item['activations'][i].float()).abs().max()) for i in layers}
        assert all(v == 0 for v in differences.values()), differences
        save_validation({'exact_activations':True, 'max_absolute_difference_by_layer':differences,
                         'stages':4, 'layers_per_stage':9, 'bounded_queue_size':2})
        del item, reference
        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(4)]
        threads.append(threading.Thread(target=producer, daemon=True))
        for thread in threads:
            thread.start()
        expected = start_batch
        while True:
            item = queues[-1].get()
            if isinstance(item, BaseException):
                raise item
            if item is None:
                break
            assert item['batch_ix'] == expected
            expected += 1
            yield item['batch_ix'], item['batch'], item['activations']
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
        for handle in handles:
            handle.remove()
