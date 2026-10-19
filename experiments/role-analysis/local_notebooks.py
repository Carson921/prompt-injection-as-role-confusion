"""Documented local adaptations; original notebook sources remain intact."""
from run_local_gptoss20b import original, OUT, HERE


def training():
    nb = original('02-train-role-probes.ipynb')
    s = nb.cells
    s[1].source = s[1].source.replace('from utils.loader import','from local_runtime import').replace(
        'from utils.probes import run_and_export_states, run_projections, check_max_seq_len',
        'from utils.probes import run_projections, check_max_seq_len\nfrom local_runtime import run_and_export_states, OUT, atomic_json\nfrom pathlib import Path\nimport hashlib, json')
    s[1].source += "\n# Bound cached CuPy allocations across all role combinations.\ncupy.get_default_memory_pool().set_limit(size=12*2**30)\n"
    s[8].source = s[8].source.replace('raw_data = load_raw_ds()',
        "raw_data = json.loads((OUT/'raw_training_data.json').read_text())\nassert len(raw_data)==249\nprint('Training source counts:',pd.DataFrame(raw_data).source.value_counts().to_dict())")
    s[11].source += "\ninput_df.to_parquet(OUT/'training_inputs.parquet',index=False)\n"
    s[12].source = s[12].source.replace('batch_size = 32','batch_size = 4')
    base14 = s[14].source
    prefix = """layers_to_probe = list(range(0,model_n_layers,2))
training_signature = hashlib.sha256(json.dumps({
    'raw_data':raw_data,'config':yaml.safe_load(open('config/probe.yaml'))[model_prefix],
    'backend':'BF16-dequantized/eager','layers':layers_to_probe,
},sort_keys=True).encode()).hexdigest()[:20]
probe_checkpoints = OUT/'probe-checkpoints'/training_signature
probe_checkpoints.mkdir(parents=True,exist_ok=True)
training_done = probe_checkpoints/'complete.json'
training_cached = training_done.exists() and len(list(probe_checkpoints.glob('*.pkl')))==96
training_activation_path = None
if training_cached:
    print('All 96 probe checkpoints found; skipping repeated activation extraction.')
    all_probe_hs = {}
else:
"""
    s[14].source = prefix + '\n'.join('    '+line for line in base14.splitlines()) + "\n    training_activation_path = res['cache_path']\n"
    s[16].source = "if training_cached:\n    probe_sample_df = pd.read_parquet(OUT/'training_labels.parquet')\nelse:\n" + '\n'.join('    '+line for line in s[16].source.splitlines()) + "\n    probe_sample_df.to_parquet(OUT/'training_labels.parquet',index=False)\n"
    s[16].source += "\n# The model is not used by logistic regression; release its GPU weights temporarily.\nmodel.to('cpu')\nclear_all_cuda_memory()\n"
    s[19].source = s[19].source.replace(
        '        probe_res = get_probe_result(',
        "        checkpoint = probe_checkpoints / (''.join(r[0] for r in roles_dict['roles'])+f'-{layer_ix}.pkl')\n        if checkpoint.exists():\n            with open(checkpoint,'rb') as f:\n                probe_res = pickle.load(f)\n        else:\n            probe_res = get_probe_result(")
    # Indent only the remaining call arguments and append an atomic checkpoint write.
    start = s[19].source.index('            probe_res = get_probe_result(')
    end = s[19].source.index('        print(f"  Layer',start)
    block = s[19].source[start:end]
    lines = block.splitlines()
    block = lines[0]+'\n'+'\n'.join('    '+line for line in lines[1:])+"\n            with open(str(checkpoint)+'.tmp','wb') as f:\n                pickle.dump(probe_res,f)\n            Path(str(checkpoint)+'.tmp').replace(checkpoint)\n"
    s[19].source = s[19].source[:start]+block+s[19].source[end:]
    s[19].source += "\nif not training_cached:\n    from local_runtime import diagnose_midlayer_probe\n    diagnose_midlayer_probe(probe_sample_df,all_probe_hs,all_probes)\n"
    s[20].source += """
atomic_json(training_done,{'signature':training_signature,'n_probes':len(all_probes)})
pd.DataFrame([{k:v for k,v in p.items() if k in ['layer_ix','role_space','acc','nll','n_inputs']} for p in all_probes]).assign(role_space=lambda d:d.role_space.map(lambda x:''.join(r[0] for r in x))).to_csv(OUT/'training_metrics.csv',index=False)
del all_probe_hs
gc.collect()
if training_activation_path:
    Path(training_activation_path).unlink()
    print('Released transient training activation map after saving all probes.')
"""
    s[23].source = "cupy.get_default_memory_pool().free_all_blocks()\ntorch.cuda.empty_cache()\nmodel.to(main_device)\n"+s[23].source
    s[25].source = s[25].source.replace('max_samples = 30 # 100 for full test','max_samples = 100 # Full test')
    s[25].source += "\natomic_json(OUT/'validation_conversations.json',convs_df.to_dict('records'))\nassert len(convs)>=11\n"
    s[29].source = s[29].source.replace('batch_size = 8','batch_size = 1')
    s[29].source += "\nconversation_activation_path = convs_outputs['cache_path']\n"
    s[30].source += "\nsample_df_labeled.to_parquet(OUT/'conversation_labels.parquet',index=False)\ninput_convs_df.drop(columns='messages').to_parquet(OUT/'conversation_inputs.parquet',index=False)\n"
    s[33].source += "\ntest_projections.to_parquet(OUT/'selected_conversation_projections.parquet',index=False)\n"
    s[36].source = """# Exact original per-conversation statistics without retaining all 96 token-level tables.
from local_runtime import aggregate_conversation_projections
probability_summary, hard_summary = aggregate_conversation_projections(all_probes,sample_df_labeled,input_convs_df,convs_hs)
"""
    s[37].source = 'out = probability_summary\n\n'+s[37].source[s[37].source.index('def draw_proj_val_table'):]
    s[38].source = """out = hard_summary
display(draw_proj_val_table(out))
out.to_csv(f'{ws}/experiments/role-analysis/outputs/probe-projections/all_conv_acc_{model_prefix}.csv',index=False)
del convs_hs, convs_outputs
gc.collect()
Path(conversation_activation_path).unlink()
print('Released transient conversation activation map.')
"""
    for i in range(40,47):
        s[i].source = "print('Cross-model comparison skipped by explicit user instruction (no external-model conversations).')" if i==40 else '# Cross-model comparison skipped by user instruction.'
    s[50].source = s[50].source.replace('max_length = 512 * 4',"max_length = check_max_seq_len(tokenizer,tomato_input_df['prompt'].tolist())").replace('batch_size = 16','batch_size = 1')
    s[54].source = s[54].source.replace('    fig.show()',"    fig.write_html(str(OUT/'plots'/f'tomato-{this_prompt_key}.html'),include_plotlyjs='cdn')\n    fig.show()")
    s[55].source += """
atomic_json(OUT/'python_workflow_complete.json',{
    'training_texts':len(raw_data),'training_sequences':len(input_df),'n_probes':len(all_probes),
    'layers':layers_to_probe,'validation_conversations':len(convs),'validation_formats':list(all_convs_by_type),
    'tomato_formats':list(tomato_prompts),'tomato_projection_rows':len(tomato_projs_df),
    'torch_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
    'torch_peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
    'cross_model_comparison':'skipped by user instruction',
})
"""
    return nb


def r_analysis():
    nb = original('03-analyze-probes.ipynb')
    nb.cells[1].source = "model_prefixes = c('gptoss-20b')"
    source = nb.cells[6].source
    start = source.index('alt_acc =')
    end = source.index('projections =',start)
    source = source[:start]+"alt_acc = base_acc[0,]\nmessage('Cross-model comparison skipped by user instruction.')\n\n"+source[end:]
    source = source.replace("model_prefixes = c('gptoss-20b', 'gptoss-120b', 'qwen3-30b-a3b', 'nemotron-3-nano')", "model_prefixes = c('gptoss-20b')")
    nb.cells[6].source = source
    nb.cells[7].source += "\nwrite_csv(convs_df,file.path(ws,'experiments/role-analysis/outputs/local-gptoss20b/r-midlayer-validation.csv'))\n"+save_r_plot('midlayer-validation','last_plot()',7,3.5)
    nb.cells[8].source += '\n'+save_r_plot('conversation-validation','plot',7,3.5)
    nb.cells[8].source += "\nwriteLines(capture.output(sessionInfo()),file.path(ws,'experiments/role-analysis/outputs/local-gptoss20b/R-sessionInfo.txt'))\n"
    return nb


def save_r_plot(name, plot='plot', width=7, height=3.5):
    base = f"{{ws}}/experiments/role-analysis/outputs/local-gptoss20b/plots/{name}"
    return f"ggsave(str_glue('{base}.png'),plot={plot},width={width},height={height},dpi=300)\nggsave(str_glue('{base}.pdf'),plot={plot},width={width},height={height},device=cairo_pdf)\n"


def r_tomato():
    nb = original('04-tomato-probe-results.ipynb')
    # Empty strings are valid reconstructed pieces of split Unicode tokens.
    nb.cells[3].source = nb.cells[3].source.replace('trim_ws = FALSE','trim_ws = FALSE, na = character()')
    for cell in nb.cells:
        cell.source = cell.source.replace('{ws}/docs/','{ws}/experiments/role-analysis/outputs/local-gptoss20b/plots/')
    # Save each line plot alongside its rendered notebook output.
    nb.cells[8].source = nb.cells[8].source.replace('    print(this_p)',
        save_r_plot('tomato-lines-{this_plot_df$prompt_key[[1]]}','this_p',7,5)+'    print(this_p)')
    for index,name in [(10,'tomato-cotness-decay025'),(11,'tomato-cotness-decay05'),(13,'tomato-cotness-two-formats')]:
        nb.cells[index].source += '\n'+save_r_plot(name,'plot',7,3.8)
    nb.cells[14].source += "\nwrite_csv(plot_df_filtered,file.path(ws,'experiments/role-analysis/outputs/local-gptoss20b/tomato-aligned-comparison.csv'))\n"
    return nb
