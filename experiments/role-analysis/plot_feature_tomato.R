# Compare the document-split baseline and selected SUCA feature model using the
# same four-panel and three-format CoT plots as the original tomato notebook.
suppressPackageStartupMessages(library(jsonlite))
ws=normalizePath('.')
here=file.path(ws,'experiments/role-analysis')
tomato_out=file.path(here,'outputs/local-gptoss120b/feature-search/tomato')
nb=fromJSON(file.path(here,'04-tomato-probe-results.ipynb'),simplifyVector=FALSE)
selection=fromJSON(file.path(here,'outputs/local-gptoss120b/feature-search/selection.json'),simplifyVector=FALSE)
source_cell=function(index)paste0(unlist(nb$cells[[index+1]]$source),collapse='')
cairo_pdf(filename='/dev/null')
for(variant in c('baseline','selected')) {
  plot_out=file.path(tomato_out,'plots',variant)
  caption=paste('GPT-OSS-120B | document split |',if(variant=='baseline') 'norm-20' else selection$document$suca$feature)
  dir.create(plot_out,recursive=TRUE,showWarnings=FALSE)
  for(index in c(1,3,5,7,14)) {
    src=source_cell(index)
    src=gsub("ws = '/workspace/deliberative-alignment-jailbreaks'","ws = normalizePath('.')",src,fixed=TRUE)
    src=gsub("model_prefix = 'gptoss-20b'","model_prefix = 'gptoss-120b-feature'",src,fixed=TRUE)
    src=gsub("file.path(ws, str_glue('experiments/role-analysis/outputs/probe-projections/tomato-role-projections-{model_prefix}.csv'))",
             "file.path(tomato_out, 'projections.csv.gz')",src,fixed=TRUE)
    src=gsub('trim_ws = FALSE)','trim_ws = FALSE, na = character()) %>% filter(model_variant == variant)',src,fixed=TRUE)
    src=gsub('layer_ix == 12',"layer_ix == 'feature-model'",src,fixed=TRUE)
    src=gsub('{ws}/experiments/role-analysis/plots/',paste0(plot_out,'/'),src,fixed=TRUE)
    src=gsub('{ws}/docs/',paste0(plot_out,'/'),src,fixed=TRUE)
    if(index==7) src=sub('    ggsave(',"    this_p = this_p + labs(caption=caption) + theme(plot.caption=element_text(size=6))\n    ggsave(",src,fixed=TRUE)
    if(index==14) src=sub('ggsave(',"plot = plot + labs(caption=caption) + theme(plot.caption=element_text(size=6))\nggsave(",src,fixed=TRUE)
    eval(parse(text=src),envir=.GlobalEnv)
  }
}
writeLines(capture.output(sessionInfo()),file.path(tomato_out,'R-sessionInfo.txt'))
dev.off()
