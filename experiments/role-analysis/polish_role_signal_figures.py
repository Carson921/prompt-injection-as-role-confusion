"""Presentation-only heatmap layout repair; frozen statistics stay unchanged."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import analyze_role_signal_factorial as analysis


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output
    frame,check=analysis.read_and_validate(out/'input-summary.csv')
    cells=frame[list(analysis.METRICS)].to_numpy().reshape(50,2,2,2,3,len(analysis.METRICS)).mean(axis=0)
    groups=[cells[0],cells[1],cells.mean(axis=0)]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(16.5,8.8),sharey=True,layout='constrained')
    labels=[f'{t.title()} · {"analysis" if w=="cot" else "user"} wrapper · {position}'
            for t in analysis.TEXTS for w in analysis.WRAPPERS for position in analysis.POSITIONS]
    for index,ax in enumerate(axes):
        values=groups[index][...,1:7].reshape(12,6)
        im=ax.imshow(values,cmap='viridis',vmin=0,vmax=1,aspect='auto')
        ax.set_xticks(range(6),('System','Developer','User','CoT','Assistant','Tool'),rotation=45,ha='right')
        ax.set_yticks(range(12),labels)
        ax.set_title(analysis.GROUP_LABELS[analysis.GROUPS[index]],pad=12)
        for r in range(12):
            for c in range(6):
                ax.text(c,r,f'{values[r,c]:.2f}',ha='center',va='center',fontsize=8,
                        color='#202020' if values[r,c]>.55 else 'white')
    fig.colorbar(im,ax=list(axes),location='right',fraction=.025,pad=.018,shrink=.8).set_label('Mean role probability')
    fig.suptitle(analysis.MODEL_TITLE+'\nAll role probabilities by condition',fontsize=13)
    analysis.save_figure(fig,out/'figures/role-signal-probabilities')
    (out/'figure-layout-manifest.json').write_text(json.dumps(dict(
        purpose='Move heatmap colorbar outside all three axes; presentation only, no statistics or frozen analysis changes.',
        source_sha256=analysis.sha256(out/'input-summary.csv'),
        layout_script_sha256=analysis.sha256(Path(__file__)),
        frozen_analysis_sha256=analysis.sha256(Path(analysis.__file__)),
        figures=['role-signal-probabilities.pdf','role-signal-probabilities.png','role-signal-probabilities.svg']),indent=2)+'\n')


if __name__=='__main__':main()
