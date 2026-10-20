"""Capture named residual/normalized features in the verified four-stage pipeline."""
from queue import Empty, Full, Queue
import threading

import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


def install_hooks(model, specs, receive):
    handles=[]
    for name, spec in specs.items():
        layer=model.model.layers[spec['layer']]
        if spec['point']=='mid':
            def pre(module,args,key=name):
                receive(key,args[0])
            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(pre))
        else:
            module=layer if spec['point']=='out' else layer.post_attention_layernorm
            def post(module,args,output,key=name):
                receive(key,output)
            handles.append(module.register_forward_hook(post))
    return handles


def feature_batches(model, dataloader, specs, start_batch, validation, save_validation):
    context=threading.local()
    stop=threading.Event()
    queues=[Queue(maxsize=2) for _ in range(5)]
    def receive(name,output):
        context.activations[name]=output.detach().to('cpu',torch.float16).reshape(-1,output.shape[-1])
    handles=install_hooks(model,specs,receive)

    def stage(index,item):
        with torch.inference_mode(),torch.cuda.device(index):
            context.activations=item['activations']
            if index==0:
                ids=item['batch']['input_ids'].to('cuda:0')
                attention=item['batch']['attention_mask'].to('cuda:0')
                hidden=model.model.embed_tokens(ids)
                positions=torch.arange(hidden.shape[1],device=hidden.device)
                position_ids=positions.unsqueeze(0)
                kwargs=dict(config=model.config,inputs_embeds=hidden,attention_mask=attention,
                            cache_position=positions,past_key_values=None)
                item.update(hidden=hidden,positions=positions,position_ids=position_ids,
                            embeddings=model.model.rotary_emb(hidden,position_ids),
                            masks={'full_attention':create_causal_mask(**kwargs),
                                   'sliding_attention':create_sliding_window_causal_mask(**kwargs)})
            hidden=item['hidden']
            for layer in model.model.layers[index*9:(index+1)*9]:
                hidden=layer(hidden,attention_mask=item['masks'][layer.attention_type],
                             position_embeddings=item['embeddings'],position_ids=item['position_ids'],
                             past_key_values=None,use_cache=False,cache_position=item['positions'])
            item['hidden']=hidden
            if index==3:
                for key in ['hidden','positions','position_ids','embeddings','masks']:
                    del item[key]
            torch.cuda.synchronize(index)
            return item

    def put(queue,value):
        while not stop.is_set():
            try:
                queue.put(value,timeout=.2)
                return
            except Full:
                pass

    def fail(error):
        put(queues[-1],error)
        stop.set()

    def producer():
        try:
            for bi,batch in enumerate(dataloader):
                if bi>=start_batch:
                    put(queues[0],dict(batch_ix=bi,batch=batch,activations={}))
                if stop.is_set():
                    return
            put(queues[0],None)
        except BaseException as error:
            fail(error)

    def worker(index):
        try:
            while not stop.is_set():
                try:
                    item=queues[index].get(timeout=.2)
                except Empty:
                    continue
                if item is None:
                    put(queues[index+1],None)
                    return
                put(queues[index+1],stage(index,item))
        except BaseException as error:
            fail(error)

    threads=[]
    try:
        sample,reference=validation
        item=dict(batch=dict(input_ids=sample.input_ids.cpu(),attention_mask=sample.attention_mask.cpu()),activations={})
        for i in range(4):
            item=stage(i,item)
        differences={name:float((reference[name].float()-item['activations'][name].float()).abs().max()) for name in specs}
        assert all(v==0 for v in differences.values()),differences
        save_validation(dict(exact_activations=True,max_absolute_difference=differences,stages=4,layers_per_stage=9))
        del item,reference
        threads=[threading.Thread(target=worker,args=(i,),daemon=True) for i in range(4)]
        threads.append(threading.Thread(target=producer,daemon=True))
        for thread in threads:
            thread.start()
        expected=start_batch
        while True:
            item=queues[-1].get()
            if isinstance(item,BaseException):
                raise item
            if item is None:
                break
            assert item['batch_ix']==expected
            expected+=1
            yield item['batch_ix'],item['batch'],item['activations']
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
        for handle in handles:
            handle.remove()
