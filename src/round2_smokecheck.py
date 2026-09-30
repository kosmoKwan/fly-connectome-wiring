import polars as pl
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)
R = '/root/fly-connectome/results/'
a = pl.read_parquet(R + 'heldout_stat_theta50.parquet').filter(pl.col('graph') == 'real')
s = pl.read_parquet(R + 'round2/smoke_pairs_scalar_theta50.parquet')
j = s.join(a.select('mode', 'run', 'window', pl.col('gmean').alias('arch')), on=['mode', 'run', 'window'])
print('PAIRS smoke rows', s.height, 'joined', j.height)
print('  max|S_all-archive|', (j['S_all'] - j['arch']).abs().max(), ' max|gmean_analyse-archive|', (j['gmean_analyse'] - j['arch']).abs().max(),
      ' max|S_all-gmean_analyse|', (j['S_all'] - j['gmean_analyse']).abs().max())
print(j.group_by('mode', 'window').agg(pl.col('S_all').mean(), pl.col('arch').mean(), pl.col('S_exact').mean(), pl.col('S_nonThC').mean(),
      pl.col('S_ThC').mean(), pl.col('S_CTr').mean(), pl.col('S_FTi').mean(), pl.col('S_TiTa').mean(), pl.col('n_gated_all').mean(), pl.col('n_gated_exact').mean()).sort('mode', 'window'))
p = pl.read_parquet(R + 'round2/smoke_pairs_pairs_theta50.parquet')
print('pair rows', p.height, 'pairs per readout', p.group_by('readout').agg(pl.col('pair').n_unique()).to_dicts())
b = pl.read_parquet(R + 'heldout_stat_manc50.parquet').filter((pl.col('graph') == 'real') & (pl.col('mode') == 'stochastic'))
l = pl.read_parquet(R + 'round2/smoke_long_scalar.parquet')
jl = l.filter(pl.col('window').str.starts_with('w')).join(b.select('run', 'window', pl.col('gmean').alias('arch')), on=['run', 'window'])
print('LONG smoke rows', l.height, 'joined', jl.height, ' max|S_all-archive|', (jl['S_all'] - jl['arch']).abs().max(), ' mean|d|', (jl['S_all'] - jl['arch']).abs().mean())
print(jl.group_by('window').agg(pl.col('S_all').mean(), pl.col('arch').mean()).sort('window'))
print(l.filter(pl.col('window').str.starts_with('t')).group_by('window').agg(pl.col('S_all').mean(), pl.col('rhythm').mean(), pl.col('n_gated_all').mean()).sort('window'))
