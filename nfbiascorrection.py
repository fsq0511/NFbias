import dask
import dask.array
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import xclim as xc
from xclim import sdba
import statsmodels.api as sm
from datetime import timedelta
from tqdm import tqdm   ##nn
from scipy.stats import multivariate_normal
import torch
import torch.utils.data as data
import zuko
import zuko.distributions as NF
import seaborn as sns
from cmip6_downscaling.methods.maca.utils import generate_batches
import hvplot.xarray
from scipy.stats import gamma
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ExponentialLR
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
import rioxarray
import geopandas as gpd
from shapely.geometry import mapping
from dask.distributed import Client, LocalCluster

def wasserstein_dbygrid(x1,x2):
    return xr.apply_ufunc(
        wasserstein_distance, x1, x2,
        input_core_dims=[['time'],['time']],
        # output_core_dims=[],
        join = 'right',
        dask='parallelized',
        vectorize=True,
        output_dtypes=['float']
    )
batches, cores = generate_batches(365, 1 , 7)
    
# cluster = LocalCluster()
# client = Client(cluster)
# client
client = Client(n_workers=10, threads_per_worker=8, memory_limit='30GB')
client
# os.chdir('/Users/FAng/Projects/CMIP6_hist')

#GFDL NorESM2 MRI MPI ACCESS
model_name = 'GFDL'
os.chdir(f'/Users/FAng/Projects/CMIP6_hist/{model_name}')
obs_file = 'Nclimdaily-'f'{model_name}-19510101_20221231.nc'
# gcm_pr_file = 'pr_day_NorESM2-LM_historical_r1i1p1f1_gn_19500101-20141231.nc' #
gcm_pr_file = 'US-'f'{model_name}_prcp_day_19500101_20141231.nc'
# gcm_tmax_file = 'tas_day_NorESM2-LM_historical_r1i1p1f1_gn_19500101-20141231.nc'  #
gcm_tmax_file ='US-'f'{model_name}_tmax_day_19500101_20141231.nc'
print(obs_file,gcm_pr_file,gcm_tmax_file)

obs_data = xr.open_dataset(obs_file,
                           chunks={ "time": -1,},
                           drop_variables=["tmin", "tavg","lon_bnds","lat_bnds"] ).sel(time =slice('1951-01-01','2014-12-31')).convert_calendar('noleap')
# obs_data = obs_data.rename({'prcp':'pr'})
obs_data['pr'] = xc.core.units.convert_units_to(obs_data.pr, "mm")
obs_data['tmax'] = xc.core.units.convert_units_to(obs_data.tmax, "K")
alldoy = obs_data.time.dt.dayofyear

gcm_data_pr = xr.open_dataset(gcm_pr_file, chunks={ "time": -1,}).sel(time =slice('1951-01-01','2014-12-31'),
                                                                      lat = obs_data.lat, lon = obs_data.lon).convert_calendar('noleap').sel(bnds =1)[[ 'time','lon','lat','pr']]
try:
    gcm_data_tmax = xr.open_dataset(gcm_tmax_file, chunks={ "time": -1,}).sel(time =slice('1951-01-01','2014-12-31')).convert_calendar('noleap').sel(bnds =1)[[ 'time','lon','lat','tasmax']]
except:
    gcm_data_tmax = xr.open_dataset(gcm_tmax_file, chunks={ "time": -1,}).sel(time =slice('1951-01-01','2014-12-31')).convert_calendar('noleap').sel(bnds =1)[[ 'time','lon','lat','tas']]


gcm_data_tmax["time"] = gcm_data_pr["time"] =obs_data["time"] 
gcm_data_tmax['lon'] = gcm_data_pr['lon'] = obs_data['lon']
gcm_data_tmax['lat'] =  gcm_data_pr['lat'] =  obs_data['lat']
gcm_data = xr.merge([gcm_data_pr,gcm_data_tmax])
try:
    gcm_data = gcm_data.rename({'tasmax':'tmax'})
except:
    gcm_data = gcm_data.rename({'tas':'tmax'})

gcm_data['pr'] = xc.core.units.convert_units_to(gcm_data.pr, "mm")

alldoy  = obs_data.time.dt.dayofyear
allmonth = obs_data.time.dt.month
# alldoy[1+365*11]

#handling zeros, add uniformed distribution data
sval2 =  np.random.uniform(0.001, 0.005, obs_data.pr.shape)
obs_data['pr1'] = obs_data.pr+sval2

obs_data['prstd'] = (obs_data.pr1 - obs_data.pr1.mean('time')) / (obs_data.pr1.std('time', ddof=1))
obs_data['tmaxstd'] = (obs_data.tmax - obs_data.tmax.mean('time')) / (obs_data.tmax.std('time', ddof=1))

gcm_data['pr1'] = gcm_data.pr+sval2

gcm_data['prstd'] = (gcm_data.pr1 - gcm_data.pr1.mean('time')) / (gcm_data.pr1.std('time', ddof=1))
gcm_data['tmaxstd'] = (gcm_data.tmax - gcm_data.tmax.mean('time')) / (gcm_data.tmax.std('time', ddof=1))


alldoy = gcm_data.time.dt.dayofyear

allmonth = gcm_data.time.dt.month
# gcm_data1 = gcm_data.sel(time =slice('1951-01-01','2014-12-31'))
qr_file = '/Users/FAng/Projects/GCM_bcdaily/'f'{model_name}/results_qr.nc'
cca_file = '/Users/FAng/Projects/GCM_bcdaily/'f'{model_name}/results_cca.nc'
qr_data = xr.open_dataset(qr_file,chunks={ "time": -1,}).rename({"pr":"pr_QR","tmax":"tmax_QR"})
cca_data = xr.open_dataset(cca_file,chunks={ "time": -1,}).rename({"pr":"pr_CCA","tmax":"tmax_CCA"})
cca_data['lon'] = qr_data['lon'] = obs_data['lon']
cca_data['lat'] =  qr_data['lat'] =  obs_data['lat']
biased_data = qr_data.combine_first(cca_data)

biased_data['time'] = gcm_data.time

gcm_data = gcm_data.combine_first(biased_data).transpose("time","lat","lon")

 alldoy = obs_data.time.dt.dayofyear

gcm_data_pr = xr.open_dataset(gcm_pr_file, chunks={ "time": -1,}).sel(time =slice('1951-01-01','2014-12-31'),
                                                                      lat = obs_data.lat, lon = obs_data.lon).convert_calendar('noleap').sel(bnds =1)[[ 'time','lon','lat','pr']]
try:
    gcm_data_tmax = xr.open_dataset(gcm_tmax_file, chunks={ "time": -1,}).sel(time =slice('1951-01-01','2014-12-31')).convert_calendar('noleap').sel(bnds =1)[[ 'time','lon','lat','tasmax']]
except:
    gcm_data_tmax = xr.open_dataset(gcm_tmax_file, chunks={ "time": -1,}).sel(time =slice('1951-01-01','2014-12-31')).convert_calendar('noleap').sel(bnds =1)[[ 'time','lon','lat','tas']]


gcm_data_tmax["time"] = gcm_data_pr["time"] =obs_data["time"] 
gcm_data_tmax['lon'] = gcm_data_pr['lon'] = obs_data['lon']
gcm_data_tmax['lat'] =  gcm_data_pr['lat'] =  obs_data['lat']
gcm_data = xr.merge([gcm_data_pr,gcm_data_tmax])
try:
    gcm_data = gcm_data.rename({'tasmax':'tmax'})
except:
    gcm_data = gcm_data.rename({'tas':'tmax'})

gcm_data['pr'] = xc.core.units.convert_units_to(gcm_data.pr, "mm")

alldoy  = obs_data.time.dt.dayofyear
allmonth = obs_data.time.dt.month
# alldoy[1+365*11]

#handling zeros, add uniformed distribution data
sval2 =  np.random.uniform(0.001, 0.005, obs_data.pr.shape)
obs_data['pr1'] = obs_data.pr+sval2

obs_data['prstd'] = (obs_data.pr1 - obs_data.pr1.mean('time')) / (obs_data.pr1.std('time', ddof=1))
obs_data['tmaxstd'] = (obs_data.tmax - obs_data.tmax.mean('time')) / (obs_data.tmax.std('time', ddof=1))

gcm_data['pr1'] = gcm_data.pr+sval2

gcm_data['prstd'] = (gcm_data.pr1 - gcm_data.pr1.mean('time')) / (gcm_data.pr1.std('time', ddof=1))
gcm_data['tmaxstd'] = (gcm_data.tmax - gcm_data.tmax.mean('time')) / (gcm_data.tmax.std('time', ddof=1))


alldoy = gcm_data.time.dt.dayofyear

allmonth = gcm_data.time.dt.month
# gcm_data1 = gcm_data.sel(time =slice('1951-01-01','2014-12-31'))
qr_file = '/Users/FAng/Projects/GCM_bcdaily/'f'{model_name}/results_qr.nc'
cca_file = '/Users/FAng/Projects/GCM_bcdaily/'f'{model_name}/results_cca.nc'
qr_data = xr.open_dataset(qr_file,chunks={ "time": -1,}).rename({"pr":"pr_QR","tmax":"tmax_QR"})
cca_data = xr.open_dataset(cca_file,chunks={ "time": -1,}).rename({"pr":"pr_CCA","tmax":"tmax_CCA"})
cca_data['lon'] = qr_data['lon'] = obs_data['lon']
cca_data['lat'] =  qr_data['lat'] =  obs_data['lat']
biased_data = qr_data.combine_first(cca_data)

biased_data['time'] = gcm_data.time

gcm_data = gcm_data.combine_first(biased_data).transpose("time","lat","lon")

 

### define joint function
def bias_normflow(x1, x2, y1, y2, v1, v2,  z1, z2):
    # flow = zuko.flows.NSF(features=2, context=2, transforms=3, hidden_features=(256,128), activation=torch.nn.ELU)
    flow = zuko.flows.MAF(features=2, context=3, transforms=3)
    # flow = zuko.flows.NSF(features=2, context=3, transforms=3)
    x1 = x1[~np.isnan(y1)] 
    x2 = x2[~np.isnan(y1)] 
    y1 = y1[~np.isnan(y1)] 
    y2 = y2[~np.isnan(y2)] 
    v1 = v1[~np.isnan(z1)] 
    v2 = v2[~np.isnan(z1)] 
    z1 = z1[~np.isnan(z1)] 
    z2 = z2[~np.isnan(z2)] 

    # print("x1",x1)
    # print("y1",y1)
    # print("v1",v1)
    # print("z1",z1)
    # flow = zuko.flows.MAF(features=2, context=2, transforms= 3)

    flow = zuko.flows.Flow(flow.transform.inv, flow.base)
    label = torch.cat((torch.zeros(len(x1)),torch.ones(len(y1))),0)

    testdatao =  pd.DataFrame(np.stack((x1, x2), axis=-1))
    testdatap =  pd.DataFrame(np.stack((y1, y2), axis=-1))
    # testdata = pd.concat([testdatao, testdatap], axis=0)
    # trainset =  torch.tensor(testdata.values , dtype=torch.float)
    trainset = torch.tensor(np.vstack((testdatao.values, testdatap.values)),dtype=torch.float)
    # trainsetc = torch.tensor(np.vstack((testdatao.values, testdatap.values))[:,1],dtype=torch.float)
    trainset0 = data.TensorDataset(trainset,torch.column_stack((trainset, label)))
    trainloader = data.DataLoader(trainset0,batch_size= 100, shuffle=True)
    # optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3, weight_decay=1e-3, betas= (0.85,0.999))
    optimizer = torch.optim.Adam(flow.parameters(), lr=2e-3)
    scheduler = ExponentialLR(optimizer, gamma=0.985)

    for epoch in range(7):
        losses = []
        for x , c in trainloader:
            # c = label.unsqueeze(dim=-1)
            loss = -flow(c).log_prob(x).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.detach())
        losses = torch.stack(losses)
        scheduler.step()

    optimizer2 = torch.optim.Adam(flow.parameters(), lr=1e-3,weight_decay=0.985)
    for epoch in range(15):
        losses = []
        for x , c in trainloader:
            # c = label.unsqueeze(dim=-1)
            loss = -flow(c).log_prob(x).mean()
            loss.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            losses.append(loss.detach())
        losses = torch.stack(losses)
        # scheduler.step()
    testdatapz =  pd.DataFrame(np.stack((z1, z2), axis= -1))

    predset = torch.tensor(testdatapz.values, dtype=torch.float)
    testdatav =  pd.DataFrame(np.stack((v1, v2), axis=-1))

    # print(";;;") 
    # conddata1 = torch.column_stack((torch.tensor(((z2)),dtype=torch.float),torch.ones(len(z1))))
    # conddata2 = torch.column_stack((torch.tensor(((v2)),dtype=torch.float),torch.zeros(len(v1))))
    conddata1 = torch.column_stack((torch.tensor(((testdatapz.values)),dtype=torch.float),torch.ones(len(z1))))
    # conddata2 = torch.column_stack((torch.tensor(((testdatav.values)),dtype=torch.float),torch.zeros(len(v1))))
    c_stars = torch.column_stack((trainset, label))
    indexs = np.linspace(start=0, stop= len(z1), num= len(z1)) #np.random.choice(c_stars.shape[0], len(z1), replace=False)  
    conddata2 = flow(c_stars[indexs]).sample((30,))
    conddata3 = torch.column_stack((conddata2.mean(axis= 0),torch.zeros(len(z1))))


    vals = flow.transform(conddata1)(predset)
    gfdl_bias = flow.transform.inv(conddata3)(vals).detach().numpy()
    return gfdl_bias[:,0],gfdl_bias[:,1]


def NF_BS(x1,x2, y1, y2,v1,v2, z1, z2):
   ds_out = xr.merge([z1, z2])
   prcp1, tmax1 =  xr.apply_ufunc(
        bias_normflow, x1, x2, y1, y2, v1,v2, z1, z2,
        input_core_dims=[['time'],['time'],['time'],['time'],['t'],['t'],['t'],['t']],
        output_core_dims=[['t'],['t']],
        # dataset_join = 'outer',
        join = 'outer',
        dask='parallelized',
        vectorize=True,
        output_dtypes=['float','float']
    )
   ds_out['prflow'] = prcp1
   ds_out['tmaxflow'] = tmax1
   ds_out=ds_out.rename({'t': 'time'})
   try:
       ds_out = ds_out.rename({'tmax':'tmax_gcm','pr':'pr_gcm'})
   except:
       ds_out = ds_out.rename({'tmaxstd':'tmax_gcm','prstd':'pr_gcm'})
   return ds_out

def NF_bias(t):
    qrbct = NF_BS(x1 = obs_data_cal.prstd.sel(time=alldoy_cal.isin(t+1)).chunk({'time': -1}),x2 = obs_data_cal.tmaxstd.sel(time=alldoy_cal.isin(t+1)).chunk({'time': -1}),
                y1 = gcm_data_cal.prstd.sel(time=alldoy_cal.isin(t+1)).chunk({'time': -1}),  y2 = gcm_data_cal.tmaxstd.sel(time=alldoy_cal.isin(t+1)).chunk({'time': -1}),
                v1 = obs_data_val.prstd.sel(t=alldoy_val.isin(t+1)).chunk({'t': -1}),  v2 = obs_data_val.tmaxstd.sel(t=alldoy_val.isin(t+1)).chunk({'t': -1}),
                z1 = gcm_data_val.prstd.sel(t=alldoy_val.isin(t+1)).chunk({'t': -1}),  z2 = gcm_data_val.tmaxstd.sel(t=alldoy_val.isin(t+1)).chunk({'t': -1}))
    return  qrbct  

def NF_bias1(t):
    qrbct = NF_BS(x1 = obs_data_cal.prstd.sel(time=alldoy_cal.isin(batches[t])).chunk({'time': -1}),x2 = obs_data_cal.tmaxstd.sel(time=alldoy_cal.isin(batches[t])).chunk({'time': -1}),
                y1 = gcm_data_cal.prstd.sel(time=alldoy_cal.isin(batches[t])).chunk({'time': -1}),  y2 = gcm_data_cal.tmaxstd.sel(time=alldoy_cal.isin(batches[t])).chunk({'time': -1}),
                v1 = obs_data_val.prstd.sel(t=alldoy_val.isin(batches[t])).chunk({'t': -1}),  v2 = obs_data_val.tmaxstd.sel(t=alldoy_val.isin(batches[t])).chunk({'t': -1}),
                z1 = gcm_data_val.prstd.sel(t=alldoy_val.isin(t+1)).chunk({'t': -1}),  z2 = gcm_data_val.tmaxstd.sel(t=alldoy_val.isin(t+1)).chunk({'t': -1}))
    return  qrbct  


def NF_biasmonth(t):
    qrbct = NF_BS(x1 = obs_data.prstd.sel(time=allmonth.isin(t+1)).chunk({'time': -1}),x2 = obs_data.tmaxstd.sel(time=allmonth.isin(t+1)).chunk({'time': -1}),
                y1 = gcm_data.prstd.sel(time=allmonth.isin(t+1)).chunk({'time': -1}),  y2 = gcm_data.tmaxstd.sel(time=allmonth.isin(t+1)).chunk({'time': -1}),
                v1 = obs_data_val.prstd.sel(time=alldoy.isin(t+1)).chunk({'time': -1}),  v2 = obs_data_val.tmaxstd.sel(time=alldoy.isin(t+1)).chunk({'time': -1}),
                z1 = gcm_data_val.prstd.sel(time=allmonth.isin(t+1)).chunk({'time': -1}),  z2 = gcm_data_val.tmaxstd.sel(time=allmonth.isin(t+1)).chunk({'time': -1}))
    return  qrbct  


def NF_biasall(t):
    qrbct = NF_BS(x1 = obs_data_cal.prstd.chunk({'time': -1}), x2 = obs_data_cal.tmaxstd.chunk({'time': -1}),
                y1 = gcm_data_cal.prstd.chunk({'time': -1}),  y2 = gcm_data_cal.tmaxstd.chunk({'time': -1}),
                v1 = obs_data_val.prstd.chunk({'t': -1}),  v2 = obs_data_val.tmaxstd.chunk({'t': -1}),
                z1 = gcm_data_val.prstd.chunk({'t': -1}),  z2 = gcm_data_val.tmaxstd.chunk({'t': -1}))
    return  qrbct  


#quick test 
gcm_data_cal = gcm_data.sel(time =slice('1951-01-01','1990-12-31')).isel(lat =slice(10,11),lon = slice(10,11)) 
obs_data_cal = obs_data.sel(time =slice('1951-01-01','1990-12-31')).isel(lat =slice(10,11),lon = slice(10,11))  
gcm_data_val = gcm_data.sel(time =slice('1991-01-01','2014-12-31')).isel(lat =slice(10,11),lon = slice(10,11)) 
obs_data_val = obs_data.sel(time =slice('1991-01-01','2014-12-31')).isel(lat =slice(10,11),lon = slice(10,11)) 
gcm_data_val = gcm_data_val.rename({'time': 't'})
obs_data_val= obs_data_val.rename({'time': 't'})
alldoy_cal = gcm_data_cal.time.dt.dayofyear
alldoy_val = obs_data_val.t.dt.dayofyear
allmonth = gcm_data_cal.time.dt.month
from joblib import Parallel, delayed
results = Parallel(n_jobs=-1, verbose=1, require='sharedmem')(delayed(NF_bias)(t) for t in tqdm(range(0,365),position=1))
result_dataset = xr.concat(results, dim='time')
result_dataset = result_dataset.transpose("time","lat","lon")
biasresults = result_dataset.compute()
biasresults
