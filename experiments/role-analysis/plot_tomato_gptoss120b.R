# Render the original 20B tomato notebook with 120B projections, preserving its
# token filters, smoothing, colors, typography, dimensions, and message labels.
# Run from the repository root after run_tomato_gptoss120b.py --stage project.
suppressPackageStartupMessages(library(jsonlite))
ws <- normalizePath('.')
here <- file.path(ws, 'experiments/role-analysis')
out <- file.path(here, 'outputs/local-gptoss120b/tomato')
notebook <- fromJSON(file.path(here, '04-tomato-probe-results.ipynb'), simplifyVector=FALSE)
get_source <- function(index) paste0(unlist(notebook$cells[[index + 1]]$source), collapse='')
# Notebook print() calls need a headless device with the same fontconfig lookup
# as the exported Cairo PDFs, rather than the base PostScript font database.
cairo_pdf(filename='/dev/null')
warning_messages <- character()

withCallingHandlers({
for (selected_layer in c(12, 16)) {
  layer_out <- file.path(here, 'outputs/local-gptoss120b/plots', paste0('layer-', selected_layer))
  dir.create(layer_out, recursive=TRUE, showWarnings=FALSE)
  saved_source <- character()
  for (index in c(1, 3, 5, 7, 8, 10, 11, 13, 14)) {
    src <- get_source(index)
    src <- gsub("ws = '/workspace/deliberative-alignment-jailbreaks'", "ws = normalizePath('.')", src, fixed=TRUE)
    src <- gsub("model_prefix = 'gptoss-20b'", "model_prefix = 'gptoss-120b'", src, fixed=TRUE)
    src <- gsub('tomato-role-projections-{model_prefix}.csv', 'tomato-role-projections-{model_prefix}.csv.gz', src, fixed=TRUE)
    # Empty token strings represent split Unicode offsets and must remain empty.
    src <- gsub('trim_ws = FALSE', 'trim_ws = FALSE, na = character()', src, fixed=TRUE)
    src <- gsub('filter(., layer_ix == 12)', paste0('filter(., layer_ix == ', selected_layer, ')'), src, fixed=TRUE)
    src <- gsub('{ws}/experiments/role-analysis/plots/', paste0(layer_out, '/'), src, fixed=TRUE)
    src <- gsub('{ws}/docs/', paste0(layer_out, '/'), src, fixed=TRUE)
    if (index == 8) {
      save_lines <- paste0(
        "ggsave(file.path(layer_out, paste0('tomato-lines-', this_plot_df$prompt_key[[1]], '.png')), plot=this_p, width=7, height=5, dpi=300)\n",
        "ggsave(file.path(layer_out, paste0('tomato-lines-', this_plot_df$prompt_key[[1]], '.pdf')), plot=this_p, width=7, height=5, device=cairo_pdf)\n",
        'print(this_p)')
      src <- gsub('print(this_p)', save_lines, src, fixed=TRUE)
    }
    saved_source <- c(saved_source, paste0('\n# Original notebook cell ', index, '\n', src))
    message('Rendering layer ', selected_layer, ', original notebook cell ', index)
    eval(parse(text=src), envir=.GlobalEnv)
    if (index %in% c(10, 11, 13)) {
      name <- switch(as.character(index), '10'='tomato-cotness-decay025',
                     '11'='tomato-cotness-decay05', '13'='tomato-cotness-two-formats')
      ggsave(file.path(layer_out, paste0(name, '.png')), plot=plot, width=7, height=3.8, dpi=300)
      ggsave(file.path(layer_out, paste0(name, '.pdf')), plot=plot, width=7, height=3.8, device=cairo_pdf)
    }
    if (index == 14) {
      write_csv(plot_df_filtered, file.path(out, paste0('aligned-comparison-layer-', selected_layer, '.csv')))
    }
  }
  writeLines(saved_source, file.path(out, paste0('adapted-notebook-layer-', selected_layer, '.R')))
}
}, warning=function(w) {
  warning_messages <<- c(warning_messages, conditionMessage(w))
  invokeRestart('muffleWarning')
})
write_csv(as.data.frame(table(warning_messages)), file.path(out, 'R-plot-warnings.csv'))
writeLines(capture.output(sessionInfo()), file.path(out, 'R-sessionInfo.txt'))
dev.off()
