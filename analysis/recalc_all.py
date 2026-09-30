"""Independent recalculation of the 60-s held-out confirmation against the pre-registered rules
in paper/draft_v0.md sections 3.23 (claim levels) and 3.26 (surgery thresholds C1-C6).
Read-only on the NAS archive. Writes CSVs to ./out/.  Claude audit, 2026-09-27."""
import polars as pl, numpy as np, os
B='/Volumes/ninishares/Experiments/fly-connectome-research-20260926/final_heldout_20260927_121507_B9ue3i/results/'
os.makedirs('out',exist_ok=True)
MAIN=['er','role','degree','leg','hl','leg_hl']; EXTRA=['degree_recip','leg_hl_recip']
WINS=['w15_25','w25_35','w35_45','w50_60']
claims=[]; surg=[]; runs=[]
for t in ['theta50','manc50']:
    df=pl.read_parquet(B+f'heldout_stat_{t}.parquet').with_columns(pl.col('gmean').fill_nan(None).alias('gm'),pl.col('freq').fill_nan(None))
    df=df.with_columns(((pl.col('osc_frac')>0.2)&(pl.col('freq')>=1)&(pl.col('freq')<=50)).fill_null(False).alias('scan_gate'))
    s=(df.filter(pl.col('mode')=='stochastic').group_by('graph','gseed','rule','window')
       .agg(pl.col('gm').mean().alias('mean_finite'),pl.col('gm').fill_null(0).mean().alias('mean_zero'),pl.col('gm').is_null().sum().alias('n_undef'),
            pl.col('rhythm').mean().alias('rhythm'),pl.col('n_gated_pairs').mean().alias('ngp'),pl.col('osc_frac').mean().alias('osc'),pl.col('freq').mean().alias('freq')))
    s.write_csv(f'out/graph_window_summary_{t}.csv')
    for w in WINS:
        sw=s.filter(pl.col('window')==w); R=sw.filter(pl.col('graph')=='real').row(0,named=True)
        fam_gt=0; allmed=True; exceed=True
        for fam in MAIN+EXTRA:
            f=sw.filter((pl.col('graph')==fam)&(pl.col('rule')=='own'))
            if f.height==0: continue
            v=f['mean_finite'].to_numpy(); med=float(np.median(v)); gt=bool((R['mean_finite']>v).all())
            if fam in MAIN: fam_gt+=gt; allmed&=(R['mean_finite']-med)>=0.15; exceed&=R['mean_finite']>med
            claims.append(dict(connectome=t,window=w,family=fam,primary=fam in MAIN,real=R['mean_finite'],fam_median=med,fam_max=float(v.max()),real_minus_median=R['mean_finite']-med,real_gt_all=gt,real_rhythm=R['rhythm'],fam_max_rhythm=float(f['rhythm'].max())))
        lvl='STRONG' if (fam_gt==6 and allmed) else ('MODERATE' if (fam_gt>=5 and exceed) else 'WEAK/FAIL')
        claims.append(dict(connectome=t,window=w,family='__LEVEL__:'+lvl,primary=True,real=R['mean_finite'],fam_median=None,fam_max=None,real_minus_median=None,real_gt_all=None,real_rhythm=R['rhythm'],fam_max_rhythm=None))
        for kind,(xg,wg) in {'plain':('mn_shuffle_joint','mn_shuffle_pool'),'strength_controlled':('mn_shuffle_joint_w','mn_shuffle_pool_w')}.items():
            for rule in ['own','real_setting']:
                X=sw.filter((pl.col('graph')==xg)&(pl.col('rule')==rule)); W=sw.filter((pl.col('graph')==wg)&(pl.col('rule')==rule))
                xv=X['mean_finite'].to_numpy(); wv=W['mean_finite'].to_numpy(); Xm=float(np.median(xv)); Wm=float(np.median(wv)); r=R['mean_finite']
                Xrh=float(np.median(X['rhythm'].to_numpy())); Wrh=float(np.median(W['rhythm'].to_numpy()))
                Xng=float(np.median(X['ngp'].to_numpy())); Xos=float(np.median(X['osc'].to_numpy()))
                c=dict(C1=Wm>=0.8*r, C2=(Xm<=0.5*r) and (r-Xm>=0.10), C3=(Xm<=0.5*Wm) and (Wm-Xm>=0.10), C4=bool((xv<r).all() and xv.max()<wv.min()),
                       C5=(Xrh>=0.8*R['rhythm']) and (Xrh>=0.8*Wrh), C6_count_proxy=(Xng>=0.8*R['ngp']) and (Xos>=0.8*R['osc']))
                surg.append(dict(connectome=t,window=w,surgery=kind,rule=rule,R=r,W_median=Wm,X_median=Xm,X_over_R=Xm/r,R_minus_X=r-Xm,W_minus_X=Wm-Xm,
                                 R_rhythm=R['rhythm'],W_rhythm=Wrh,X_rhythm=Xrh,R_pairs=R['ngp'],X_pairs=Xng,R_osc=R['osc'],X_osc=Xos,**c,
                                 all_six=all(c.values()),collapse_def=(Xm<=0.05 and c['C1'] and c['C5'] and c['C6_count_proxy']),codex_2cond=(Xm<=0.5*r and Wm-Xm>=0.10)))
    runs.append(df.filter(pl.col('graph')=='real').select('connectome','mode','run','window','gm','rhythm','freq','osc_frac','n_gated_pairs','plv_anti','scan_gate'))
pl.DataFrame(claims).write_csv('out/claim_levels_3_23_on_heldout.csv')
pl.DataFrame(surg).write_csv('out/surgery_conditions_3_26_on_heldout.csv')
pl.concat(runs).rename({'gm':'gmean'}).write_csv('out/real_graph_all_runs.csv')
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250)
print(pl.DataFrame(claims).filter(pl.col('family').str.starts_with('__')).select('connectome','window','family','real'))
print(pl.DataFrame(surg).filter(pl.col('window').is_in(['w15_25','w50_60'])).select('connectome','window','surgery','rule','X_over_R','R_minus_X','W_minus_X','C1','C2','C3','C4','C5','C6_count_proxy','all_six','collapse_def','codex_2cond'))
