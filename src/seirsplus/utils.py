# External Libraries
import numpy as np
import pandas as pd
import json
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources
# Internal Libraries
from seirsplus.models import configs


def load_config(filename):
    try:
        with pkg_resources.open_text(configs, filename) as cfg_file:
            if('.json' in filename):
                cfg_dict = json.load(cfg_file)
                return cfg_dict
            elif('.csv' in filename):
                cfg_array = np.genfromtxt(cfg_file, delimiter=',')
                return cfg_array
            elif('.tsv' in filename):
                cfg_array = np.genfromtxt(cfg_file, delimiter='\t')
                return cfg_array
            else:
                print("load_config error: File type not supported (support for .json, .csv, .tsv)")
                exit()
    except FileNotFoundError:
        print("load_config error: Config file \'"+filename+"\' not found in seirsplus.models.configs.")
        exit()


def save_dataframe(df, file_name, file_extn='.csv', reduce_dtypes=True, flag_col_to_numeric=True, attrib_col_to_numeric=True,
                    sep=',', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, mode='w', 
                    encoding=None, compression='infer', quoting=None, quotechar='"', line_terminator=None, chunksize=None, 
                    date_format=None, doublequote=True, escapechar=None, decimal='.', errors='strict'):
    if(reduce_dtypes):
        for (column) in df:
            if('_flag_' in column and flag_col_to_numeric):
                df[column] = pd.to_numeric(df[column], downcast='integer', errors='coerce')
            if('_attrib_' in column and attrib_col_to_numeric):
                df[column] = pd.to_numeric(df[column], downcast='float', errors='coerce')
            if(df[column].dtype in [np.int64, np.int32, np.int16, np.int8]):
                if(df[column].min() >= 0):
                    df[column] = pd.to_numeric(df[column], downcast='signed', errors='coerce') 
                else:
                    df[column] = pd.to_numeric(df[column], downcast='integer', errors='coerce')
            elif(df[column].dtype in [np.float64, np.float32, np.float16]):
                df[column] = pd.to_numeric(df[column], downcast='float', errors='coerce') # downcast arg uses smallest possible int that can hold the values
    df.to_csv(path_or_buf=file_name+file_extn, sep=',', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, 
                    mode='w', encoding=None, compression='infer', quoting=None, quotechar='"', line_terminator=None, chunksize=None, 
                    date_format=None, doublequote=True, escapechar=None, decimal='.', errors='strict')


def treat_as_list(val):
    if(not isinstance(val, (list, np.ndarray)) and val is not None):
        return [val]
    elif(isinstance(val, (np.ndarray))):
        return val.flatten()
    else:
        return val

def param_as_array(param, shape):
    return np.array(param).reshape(shape) if isinstance(param, (list, np.ndarray)) else np.full(fill_value=param, shape=shape)


def param_as_bool_array(param, n, shape=None, selection_mode='binomial'):
    shape = (1, n) if shape is None else shape
    if(isinstance(param, (int, float)) and param>=0 and param<=1):
        if(selection_mode == 'binomial'):
            return np.array(param_as_array(np.random.binomial(n=1, p=param, size=n), shape), dtype=bool).flatten()
        elif(selection_mode == 'choice'):
            selected_inds = np.random.choice(range(n), size=int(param*n), replace=False)
            return param_as_bool_array([True if i in selected_inds else False for i in range(n)], n, shape).flatten()   
    else:
        return np.array(param_as_array(param, shape), dtype=bool).flatten()


def gamma_dist(mean, coeffvar, N):
    scale = mean * coeffvar ** 2
    shape = mean / scale
    return np.random.gamma(scale=scale, shape=shape, size=N)


# def dist_stats(dists, names=None, plot=False, bin_size=1, colors=None, reverse_plot=False):
def dist_stats(dists, names=None, print_stats=False):
    dists  = [dists] if not isinstance(dists, list) else dists
    names  = ([names] if (names is not None and not isinstance(names, list)) else (names if names is not None else [None] * len(dists)))
    # colors = ([colors] if (colors is not None and not isinstance(colors, list)) else (colors if colors is not None else plt.rcParams["axes.prop_cycle"].by_key()["color"]))

    stats = {}

    for i, (dist, name) in enumerate(zip(dists, names)):

        stats.update(
            {
                (name if name is not None else 'dist'+str(i+1)) + "_mean": np.mean(dist),
                (name if name is not None else 'dist'+str(i+1)) + "_stdev": np.std(dist),
                (name if name is not None else 'dist'+str(i+1)) + "_95CI": (np.percentile(dist, 2.5), np.percentile(dist, 97.5)),
            }
        )

        if(print_stats):
            print(
                (name if name is not None else 'dist'+str(i+1)+':')
                + " mean = %.2f, std = %.2f, 95%% CI = (%.2f, %.2f)"
                % (
                    np.mean(dist),
                    np.std(dist),
                    np.percentile(dist, 2.5),
                    np.percentile(dist, 97.5),
                )
            )
        # print()

        # if plot:
        #     plt.hist(
        #         dist,
        #         bins=np.arange(0, int(max(dist) + 1), step=bin_size),
        #         label=(name if name else False),
        #         color=colors[i],
        #         edgecolor="white",
        #         alpha=0.6,
        #         zorder=(-1 * i if reverse_plot else i),
        #     )

    # if plot:
    #     plt.ylabel("num nodes")
    #     plt.legend(loc="upper right")
    #     plt.show()

    return stats


# def network_stats(networks, names=None, calc_avg_path_length=False, calc_num_connected_comps=False, plot=False, bin_size=1, colors=None, reverse_plot=False):
def network_stats(networks, names=None, calc_avg_path_length=False, calc_connected_components=False):
    import networkx
    networks = [networks] if not isinstance(networks, list) else networks
    names    = ['']*len(networks) if names is None else [names] if not isinstance(names, list) else names
    # colors = [colors] if(colors is not None and not isinstance(colors, list)) else (colors if colors is not None else plt.rcParams['axes.prop_cycle'].by_key()['color'])
    
    stats = {}

    for i, (network, name) in enumerate(zip(networks, names)):
    
        degree        = [d[1] for d in network.degree()]

        degree_mean   = np.mean(degree)
        degree_stdev  = np.std(degree)
        degree_95CI   = (np.percentile(degree, 2.5), np.percentile(degree, 97.5))
        degree_CV     = np.std(degree)/np.mean(degree)
        degree_CV2    = (np.std(degree)/np.mean(degree))**2

        # print(degree_mean, "(", np.percentile(degree, 25), np.median(degree), np.percentile(degree, 75), ")")

        try: 
            assortativity   = networkx.degree_assortativity_coefficient(network)  
        except: 
            assortativity   = None
        
        try: 
            cluster_coeff   = networkx.average_clustering(network)                
        except: 
            cluster_coeff   = None
        
        if(calc_avg_path_length):
            try: 
                avg_path_length = networkx.average_shortest_path_length(network)      
            except: 
                avg_path_length = None
        else:
            avg_path_length = None

        # if(calc_num_connected_comps):
        #     try:
        #         num_connected_comps = networkx.algorithms.components.number_connected_components(network)
        #     except:
        #         num_connected_comps = None
        # else:
        #     num_connected_comps = None
        
        if(calc_connected_components):
            try:
                connected_components      = sorted(networkx.connected_components(network), key=len, reverse=True)
                num_connected_components  = len(connected_components)
                connected_component_sizes = [len(comp) for comp in connected_components]
            except:
                num_connected_components  = None
                connected_component_sizes = None
        else:
            num_connected_components  = None
            connected_component_sizes = None

        stats.update( { (name+'_' if len(name)>0 else '')+'degree_mean':          degree_mean,
                        (name+'_' if len(name)>0 else '')+'degree_stdev':         degree_stdev,
                        (name+'_' if len(name)>0 else '')+'degree_95CI':          degree_95CI,
                        (name+'_' if len(name)>0 else '')+'degree_CV':            degree_CV,
                        (name+'_' if len(name)>0 else '')+'degree_CV2':           degree_CV2,
                        (name+'_' if len(name)>0 else '')+'assortativity':        assortativity,
                        (name+'_' if len(name)>0 else '')+'cluster_coeff':        cluster_coeff,
                        (name+'_' if len(name)>0 else '')+'avg_path_length':      avg_path_length,
                        (name+'_' if len(name)>0 else '')+'num_connected_components':  num_connected_components,
                        (name+'_' if len(name)>0 else '')+'mean_connected_component_size':  np.mean(connected_component_sizes) if connected_component_sizes != None else None,
                        (name+'_' if len(name)>0 else '')+'median_connected_component_size':  np.median(connected_component_sizes) if connected_component_sizes != None else None,
                        (name+'_' if len(name)>0 else '')+'min_connected_component_size':  np.min(connected_component_sizes) if connected_component_sizes != None else None,
                        (name+'_' if len(name)>0 else '')+'max_connected_component_size':  np.max(connected_component_sizes) if connected_component_sizes != None else None  } )

        # if(name):
        #     print(name+":")
        # print("Degree: mean = %.2f, std = %.2f, 95%% CI = (%.2f, %.2f)\n        CV = %.2f, CV^2 = %.2f" 
        #       % (degree_mean, degree_stdev, degree_95CI[0], degree_95CI[1], degree_CV, degree_CV2) ) 
        # print("Assortativity:    ", assortativity)
        # print("Clustering coeff: ", cluster_coeff)
        # print("Avg. path length: ", avg_path_length)
        # print("Connected comps.: ", num_connected_comps)
        
    #     if(plot):
    #         plt.hist(degree, bins=np.arange(0, int(max(degree)+1), step=bin_size), label=(name+" degree" if name else False), color=colors[i], edgecolor='white', alpha=0.6, zorder=(-1*i if reverse_plot else i))
    
    # if(plot):
    #     plt.ylabel('num nodes')
    #     plt.legend(loc='upper right')
    #     plt.show()

    return stats



def union_of_networks(networks):
    import networkx
    return networkx.compose_all(networks)






