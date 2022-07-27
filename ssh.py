""" transfer files between local machine and remote server"""
import paramiko
import os 
import numpy as np 

client = paramiko.SSHClient()
client.load_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
# client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

client.connect('ganxis3.nest.rpi.edu', username='mac6', password='woods*score&sister')

def transfer_files(directory, filenames):

    server_des = '/home/mac6/RPI/research/timedelay/data/' 
    local_des = '/home/mac/RPI/research/timedelay/data/'
    if not os.path.exists(local_des):
        os.makedirs(local_des)
    sftp = client.open_sftp()
    if '/' in directory:
        if not os.path.exists(local_des + directory):
            os.makedirs(local_des + directory)
        filenames = sftp.listdir(server_des+directory) 
    for i in filenames:
        sftp.get(server_des + directory + i, local_des + directory +i)
        #sftp.put(local_des + directory +i, server_des + directory + i)
    sftp.close()

dynamics = 'CW'
dynamics = 'SIS'
dynamics = 'harvest'
dynamics = 'genereg'
dynamics = 'BDP'
dynamics = 'PPI'
dynamics = 'mutual'
network_type_list = ['ER', 'SF', 'RR']
network_type_list = ['SF']
N_list = [1000]

d_RR = [4, 8, 16]
d_SF = [[2.5, 99, 3], [3, 99, 3], [3.5, 99, 3], [4, 99, 3]]
d_SF = [[2.5, 999, 3]]
d_SF = [[2.5, 999, 3], [3, 999, 4], [3.8, 999, 5]]
d_ER = [2000]
d_ER = [2000, 4000, 8000]
betaeffect = 0
beta_list = [0.5]
beta_list = [0.01, 0.1, 1]
directory =  '../data/'
if betaeffect:
    beta_wt = 'beta'
else:
    beta_wt = 'wt'

for network_type in network_type_list:
    for beta in beta_list:
        for N in N_list:
            if network_type == 'SF':
                d_list = d_SF
            elif network_type == 'ER':
                d_list = d_ER
            elif network_type == 'RR':
                d_list = d_RR
            elif network_type == '2D':
                d_list = [4]

            for d in d_list:
                des = directory + dynamics + '/' + network_type + '/tau_multi/' 
                if not os.path.exists(des):
                    os.makedirs(des)
                filenames1 = des + f'N={N}_d=' + str(d) + '_' + beta_wt + '=' + str(beta) + '_logistic.csv'
                filenames2 = des + f'N={N}_d=' + str(d) +'_' +  beta_wt + '=' + str(beta) + '_decouple_two_logistic.csv'
                filenames3 = des + f'N={N}_d=' + str(d) +'_' +  beta_wt + '=' + str(beta) + '_eigenvalue_tau.csv'
                filenames4 = des + f'N={N}_d=' + str(d) +'_' +  beta_wt + '=' + str(beta) + '_decouple_eff_logistic.csv'
                filenames5 = des + f'N={N}_d=' + str(d) +'_' +  beta_wt + '=' + str(beta) + '_eigen_decouple.csv'
                des = directory + dynamics + '/' + network_type + '/tau_evo/'
                if not os.path.exists(des):
                    os.makedirs(des)
                filenames6 = des+ f'N={N}_d=' + str(d) +'_' +  beta_wt + '=' + str(beta) + '_tau_evolution.csv'
                filenames7 = des+ f'N={N}_d=' + str(d) +'_' +  beta_wt + '=' + str(beta) + '_tau_RK.csv'
                filenames8 = des+ f'N={N}_d=' + str(d) +'_' +  beta_wt + '=' + str(beta) + '_tau_Euler.csv'
                des = directory + dynamics + '/' + network_type + '/tau_model/'
                if not os.path.exists(des):
                    os.makedirs(des)
                filenames9 = des+ f'N={N}_d=' + str(d) +'_' +  beta_wt + '=' + str(beta) + '_tau_separate.csv'

                #transfer_files('', [filenames5])

dynamics = 'harvest'
tau_list = [1.0, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0, 5.0, 10.0]
initial_condition_list = [0.1, 1.0, 6.0, 7.0, 10.0]

dynamics = 'mutual'
tau_list = [0.025, 0.1, 0.15, 0.2, 0.3]
initial_condition_list = [0.1, 5.0, 6.0, 10.0]


for tau in tau_list:
    for initial_condition in initial_condition_list:
        des = '../data/' + dynamics + f'/single/tau={tau}/'
        if not os.path.exists(des):
            os.makedirs(des)
        filenames = des + f'x0={initial_condition}.csv'
        #transfer_files('', [filenames])

dynamics  = 'mutual'
network_type = 'SF'
network_type = 'ER'
N = 5000
d = [3, 99, 3]
beta_list = np.arange(1, 10, 0.5)
betaeffect = 1
delay = 0.3
initial_value = 5.0
d_list = [[2.5, 99, 3], [3, 99, 3], [3.5, 99, 3], [4, 99, 3]]
d_list = [[2.5, 999, 3]]
d_list = np.array([2, 4, 8, 16]) * N
d_list = [40000]
beta_list = [5.1, 5.2, 5.3, 5.4]
delay_list = [0.2]
for d in d_list:
    for delay in delay_list:
        for beta in beta_list:
            des = '../data/' + dynamics + '/' + network_type + '/xs/' 
            if not os.path.exists(des):
                os.makedirs(des)
            if betaeffect == 0:
                filenames = des + f'N={N}_d={d}_wt={beta}_delay={delay}_x0={initial_value}.csv'
            else:
                filenames = des + f'N={N}_d={d}_beta={beta}_delay={delay}_x0={initial_value}.csv'
            #transfer_files('', [filenames])

dynamics = 'BDP'
dynamics = 'mutual'
dynamics = 'PPI'
network_type_list = ['SF']
d_SF = [[2.5, 999, 3], [3, 999, 4], [3.8, 999, 5]]
network_type_list = ['ER']
d_ER = [2000, 4000, 8000]
beta_list = [0.01, 0.1, 1]
N_list = [1000]
blocks_list = np.arange(1, 11, 1)
for network_type in network_type_list:
    for beta in beta_list:
        for N in N_list:
            if network_type == 'SF':
                d_list = d_SF
            elif network_type == 'ER':
                d_list = d_ER
            elif network_type == 'RR':
                d_list = d_RR
            elif network_type == '2D':
                d_list = [4]

            for d in d_list:
                des = directory + dynamics + '/' + network_type + '/eigen_blocks/' 
                if not os.path.exists(des):
                    os.makedirs(des)
                for group_num in blocks_list:
                    filenames1 = des + f'N={N}_d=' + str(d) + '_' + beta_wt + '=' + str(beta) + f'_group_num={group_num}.csv'
                    #transfer_files('', [filenames1])


N = 1000
beta = 1
dynamics_list = ['mutual', 'BDP', 'PPI']
network_type_list = ['RGG']
d_list = [0.07]
d_list = [0.04, 0.05, 0.07]
r_list = [0.01, 0.05, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1]
group_num_list = np.arange(1, 10, 1).tolist()
seed_list = np.arange(10).tolist()
for dynamics in dynamics_list:
    for network_type in network_type_list:
        for d in d_list:
            des = '../data/' + dynamics + '/' + network_type + '/xs_group_decouple_knn/'
            if not os.path.exists(des):
                os.makedirs(des)
            for r in r_list:
                for group_num in group_num_list:
                    for seed in seed_list:
                        filenames1 = des + f'N={N}_d=' + str(d) + '_' + beta_wt + '=' + str(beta) + f'_r={r}_group_num={group_num}_seed={seed}.csv'
                        #transfer_files('', [filenames1])

N = 1000
beta = 0.15
dynamics_list = ['mutual']
network_type_list = ['SF']
d_list = [[2.1, 999, 2]]
group_num_list = np.arange(1, 20, 1).tolist()
seed1 = np.arange(10).tolist()
seed_list = seed1
seed_list = np.vstack((seed1, seed1)).transpose().tolist() 
iteration_step = 10
space = 'log'


for dynamics in dynamics_list:
    for network_type in network_type_list:
        for d in d_list:
            des = '../data/' + dynamics + '/' + network_type + f'/xs_group_iteraction_{iteration_step}_' + space + '/'
            if not os.path.exists(des):
                os.makedirs(des)
            for group_num in group_num_list:
                for seed in seed_list:
                    filenames1 = des + f'N={N}_d=' + str(d) + '_' + beta_wt + '=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
                    #transfer_files('', [filenames1])

N = 1000
beta = 0.1
dynamics_list = ['mutual']
network_type_list = ['SF']
d_list = [[2.5, 999, 3]]
group_num_list = np.arange(1, 20, 1).tolist()
seed1 = np.arange(10).tolist()
seed_list = seed1
seed_list = np.vstack((seed1, seed1)).transpose().tolist() 
iteration_step = 10
space = 'log'


for dynamics in dynamics_list:
    for network_type in network_type_list:
        for d in d_list:
            des = '../data/' + dynamics + '/' + network_type + f'/xs_beta_iteraction_{iteration_step}_' + space + '/'
            if not os.path.exists(des):
                os.makedirs(des)
            for group_num in group_num_list:
                for seed in seed_list:
                    filenames1 = des + f'N={N}_d=' + str(d) + '_' + beta_wt + '=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
                    #transfer_files('', [filenames1])

N = 1000
beta = 0.5
dynamics_list = ['mutual']
network_type_list = ['SF']
d_list = [[2.5, 999, 3]]
group_num_list = np.arange(1, 30, 1).tolist()
seed1 = np.arange(10).tolist()
seed_list = seed1
seed_list = np.vstack((seed1, seed1)).transpose().tolist() 
iteration_step = 10
space = 'log'


for dynamics in dynamics_list:
    for network_type in network_type_list:
        for d in d_list:
            des = '../data/' + dynamics + '/' + network_type + f'/xs_beta_two_cluster_iteraction_{iteration_step}_' + space + '/'
            if not os.path.exists(des):
                os.makedirs(des)
            for group_num in group_num_list:
                for seed in seed_list:
                    filenames1 = des + f'N={N}_d=' + str(d) + '_' + beta_wt + '=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
                    #transfer_files('', [filenames1])

N = 1000
beta = 0.5
dynamics_list = ['mutual']
network_type_list = ['SF']
d_list = [[2.5, 999, 3]]
group_num_list = np.arange(1, 20, 1).tolist()
seed1 = np.arange(10).tolist()
seed_list = seed1
seed_list = np.vstack((seed1, seed1)).transpose().tolist() 
iteration_step = 10
space = 'log'


for dynamics in dynamics_list:
    for network_type in network_type_list:
        for d in d_list:
            des = '../data/' + dynamics + '/' + network_type + f'/xs_beta_two_cluster_iteraction_{iteration_step}_adaptive_' + space + '/'
            if not os.path.exists(des):
                os.makedirs(des)
            for group_num in group_num_list:
                for seed in seed_list:
                    filenames1 = des + f'N={N}_d=' + str(d) + '_' + beta_wt + '=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
                    #transfer_files('', [filenames1])



dynamics = 'CW_high'
dynamics = 'CW'
dynamics = 'genereg'
dynamics = 'mutual'

network_type = 'SF'
network_type = 'ER'
des = '../data/' + dynamics + '/' + network_type + '/' + 'xs_bifurcation/xs_multi/' 
#transfer_files(des, [])
des = '../data/' + dynamics + '/' + network_type + '/' + 'xs_bifurcation/degree_kmeans_space=log/' 
#transfer_files(des, [])

network_type = 'SF'
network_type = 'ER'
des = '../data/' + dynamics + '/' + network_type + '/' + 'xs_bifurcation/wc_multi_rdw=0.01/'
des = '../data/' + dynamics + '/' + network_type + '/' + 'xs_bifurcation/wc_group_rdw=0.01/'
#transfer_files(des, [])


network_type = 'SF'
N = 1000
d_list = [[2.5, 999, 3]]
m_list = np.unique(np.array(np.round([(2**0.25) ** i for i in range(40)], 0), int) )
seed_list = [[i, i] for i in range(10)]

des_xs_group =  dynamics + '/' + network_type + '/' + 'xs_bifurcation/degree_kmeans_space=log_rdw=0.01/'
des_xs_multi =  dynamics + '/' + network_type + '/' + 'xs_bifurcation/xs_multi_rdw=0.01/'
#transfer_files('', [ des_xs_group + f'N={N}_d={d}_number_groups={m}_seed={seed}.csv' for d in d_list for m in m_list for seed in seed_list])
#transfer_files('', [ des_xs_multi + f'N={N}_d={d}_seed={seed}.csv' for d in d_list for seed in seed_list])

des_ygl_error =  dynamics + '/' + network_type + '/' + 'xs_bifurcation/ygl_error/'
#transfer_files(des_ygl_error, [])

dynamics = 'mutual'
dynamics = 'genereg'
dynamics = 'CW_high'
dynamics = 'CW'
des_y_multi =  dynamics + '/' + network_type + '/' + 'xs_bifurcation/y_multi_beta/'
des_y_group =  dynamics + '/' + network_type + '/' + 'xs_bifurcation/y_group_beta/'

transfer_files(des_y_multi, [])
transfer_files(des_y_group, [])

des_ygl =  dynamics + '/' + network_type + '/' + 'xs_bifurcation/ygl_beta/'
#transfer_files(des_ygl, [])

