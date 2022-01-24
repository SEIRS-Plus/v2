# External Libraries
import numpy
import matplotlib.pyplot as pyplot


def gamma_dist(mean, coeffvar, N):
    scale = mean * coeffvar ** 2
    shape = mean / scale
    return numpy.random.gamma(scale=scale, shape=shape, size=N)


def dist_stats(
    dists, names=None, plot=False, bin_size=1, colors=None, reverse_plot=False
):
    dists = [dists] if not isinstance(dists, list) else dists
    names = (
        [names]
        if (names is not None and not isinstance(names, list))
        else (names if names is not None else [None] * len(dists))
    )
    colors = (
        [colors]
        if (colors is not None and not isinstance(colors, list))
        else (
            colors
            if colors is not None
            else pyplot.rcParams["axes.prop_cycle"].by_key()["color"]
        )
    )

    stats = {}

    for i, (dist, name) in enumerate(zip(dists, names)):

        stats.update(
            {
                name + "_mean": numpy.mean(dist),
                name + "_stdev": numpy.std(dist),
                name
                + "_95CI": (numpy.percentile(dist, 2.5), numpy.percentile(dist, 97.5)),
            }
        )

        print(
            (name + ": " if name else "")
            + " mean = %.2f, std = %.2f, 95%% CI = (%.2f, %.2f)"
            % (
                numpy.mean(dist),
                numpy.std(dist),
                numpy.percentile(dist, 2.5),
                numpy.percentile(dist, 97.5),
            )
        )
        print()

        if plot:
            pyplot.hist(
                dist,
                bins=numpy.arange(0, int(max(dist) + 1), step=bin_size),
                label=(name if name else False),
                color=colors[i],
                edgecolor="white",
                alpha=0.6,
                zorder=(-1 * i if reverse_plot else i),
            )

    if plot:
        pyplot.ylabel("num nodes")
        pyplot.legend(loc="upper right")
        pyplot.show()

    return stats


def network_stats(networks, names=None, calc_avg_path_length=False, calc_num_connected_comps=False, plot=False, bin_size=1, colors=None, reverse_plot=False):
    import networkx
    networks = [networks] if not isinstance(networks, list) else networks
    names    = ['']*len(networks) if names is None else [names] if not isinstance(names, list) else names
    colors = [colors] if(colors is not None and not isinstance(colors, list)) else (colors if colors is not None else pyplot.rcParams['axes.prop_cycle'].by_key()['color'])
    
    stats = {}

    for i, (network, name) in enumerate(zip(networks, names)):
    
        degree        = [d[1] for d in network.degree()]

        degree_mean   = numpy.mean(degree)
        degree_stdev  = numpy.std(degree)
        degree_95CI   = (numpy.percentile(degree, 2.5), numpy.percentile(degree, 97.5))
        degree_CV     = numpy.std(degree)/numpy.mean(degree)
        degree_CV2    = (numpy.std(degree)/numpy.mean(degree))**2

        print(degree_mean, "(", numpy.percentile(degree, 25), numpy.median(degree), numpy.percentile(degree, 75), ")")

        try: 
            assortativity   = networkx.degree_assortativity_coefficient(network)  
        except: 
            assortativity   = numpy.nan
        
        try: 
            cluster_coeff   = networkx.average_clustering(network)                
        except: 
            cluster_coeff   = numpy.nan
        
        if(calc_avg_path_length):
            try: 
                avg_path_length = networkx.average_shortest_path_length(network)      
            except: 
                avg_path_length = numpy.nan
        else:
            avg_path_length = numpy.nan

        if(calc_num_connected_comps):
            try:
                num_connected_comps = networkx.algorithms.components.number_connected_components(network)
            except:
                num_connected_comps = numpy.nan
        else:
            num_connected_comps = numpy.nan

        stats.update( { 'degree_mean'+('_'+name if len(name)>0 else ''):          degree_mean,
                        'degree_stdev'+('_'+name if len(name)>0 else ''):         degree_stdev,
                        'degree_95CI'+('_'+name if len(name)>0 else ''):          degree_95CI,
                        'degree_CV'+('_'+name if len(name)>0 else ''):            degree_CV,
                        'degree_CV2'+('_'+name if len(name)>0 else ''):           degree_CV2,
                        'assortativity'+('_'+name if len(name)>0 else ''):        assortativity,
                        'cluster_coeff'+('_'+name if len(name)>0 else ''):        cluster_coeff,
                        'avg_path_length'+('_'+name if len(name)>0 else ''):      avg_path_length,
                        'num_connected_comps'+('_'+name if len(name)>0 else ''):  num_connected_comps  } )

        if(name):
            print(name+":")
        print("Degree: mean = %.2f, std = %.2f, 95%% CI = (%.2f, %.2f)\n        CV = %.2f, CV^2 = %.2f" 
              % (degree_mean, degree_stdev, degree_95CI[0], degree_95CI[1], degree_CV, degree_CV2) ) 
        print("Assortativity:    ", assortativity)
        print("Clustering coeff: ", cluster_coeff)
        print("Avg. path length: ", avg_path_length)
        print("Connected comps.: ", num_connected_comps)
        
        if(plot):
            pyplot.hist(degree, bins=numpy.arange(0, int(max(degree)+1), step=bin_size), label=(name+" degree" if name else False), color=colors[i], edgecolor='white', alpha=0.6, zorder=(-1*i if reverse_plot else i))
    
    if(plot):
        pyplot.ylabel('num nodes')
        pyplot.legend(loc='upper right')
        pyplot.show()

    return stats
