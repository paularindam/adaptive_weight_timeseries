import os
import pandas as pd
import json
import numpy as np
from collections import deque
from tensorboardX import SummaryWriter
import torch



class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, deque):
            return 'deque'
        return json.JSONEncoder.default(self, obj)


class SummaryLogger(object):
    """Writes entries directly to event files in the logdir

    """
    def __init__(self, logdir=None, obj = None, env = None, comment=''):

        if not logdir:
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            logdir = os.path.join(
                'runs', current_time + '_' + socket.gethostname() + comment)
            os.makedirs(logdir)
        else:
            try:
                os.makedirs(logdir)
            except FileExistsError:
                pass

        self.logdir = logdir
        # self.writer = SummaryWriter(self.logdir)
        self.writer = None
        self.scalar_dict = {}
        self.vector_dict = {}

        if obj is not None:
            self._create_param_file(obj, env)

    def _create_param_file(self, agent, env):
        my_dict = {}
        for key, item in agent.__dict__.items():
            my_dict[key] = item
            if (key == 'pnetwork') or (key == 'optimizer'):
                my_dict[key] = str(item)
            elif key == 'noise':
                my_dict[key] = item.__dict__
            elif key == 'memory':
                test_dict = {k:v for (k,v) in item.__dict__.items() if k is not 'experience'}
                my_dict[key] = test_dict
        with open(self.logdir+'/full-params.json', 'w') as outfile:
            data = json.dump(my_dict, outfile, cls=MyEncoder, indent=4)
            if env is not None:
                outfile.write('\n')
                env_dict = {}
                for key, item in env.__dict__.items():
                    env_dict[key] = item
                    if (key == 'data') or (key == 'np_random'):
                        env_dict[key] = None
                json.dump(env_dict, outfile, cls=MyEncoder, indent=4)
            file = open(self.logdir+'/network_summary.txt','w') 
        file.write(str(agent.pnetwork)) 
        file.close() 



    def __append_to_scalar_dict(self, tag, scalar_value, global_step, timestep):
        from tensorboardX.x2num import make_np
        if tag not in self.scalar_dict.keys():
            self.scalar_dict[tag] = []
        self.scalar_dict[tag].append(
            [global_step, timestep, float(make_np(scalar_value))])


    def __append_to_vector_dict(self, tag, vector_value, global_step, timestep):
        import numpy as np
        if tag not in self.vector_dict.keys():
            self.vector_dict[tag] = []
        self.vector_dict[tag].append(
            [global_step, timestep, np.array(vector_value)])

    def add_scalar(self, tag, scalar_value, global_step=None, timestep=None, tb=False):
        self.__append_to_scalar_dict(tag, scalar_value, global_step, timestep)
        if tb is True:
            self.writer.add_scalar(tag, scalar_value, global_step)

    def add_scalars(self,tag, dict_scalars, global_step=None, timestep=None, tb=False):
        if tb is True:
            self.writer.add_scalars(tag, dict_scalars, global_step)


    def add_custom_scalar(self, tag, scalar_value, global_step=None, timestep=None, tb=False):
        if tb is True:
            self.writer.add_scalar(tag, scalar_value, global_step)


    def add_vector(self, tag, vector_value, global_step=None, timestep=None):
        self.__append_to_vector_dict(tag, vector_value, global_step, timestep)

    def add_histogram(self, tag, agent, _iter):
        for name, module in agent.named_children():
            self.writer.add_histogram('network/'+tag+'_'+name+'_w', module.weight, _iter)
            self.writer.add_histogram('network/'+tag+'_'+name+'_b', module.bias, _iter)

    def save_model(self, network, tag='checkpoint_actor.pth'):
        print('saving model')
        torch.save(network.state_dict(), self.logdir+'/'+tag)


    def __create_target_filename(self, key):
        filename = key + '.csv'
        return os.path.join(self.logdir,filename)


    def dict_to_files(self):
        if self.scalar_dict is not None:
            for key, value in self.scalar_dict.items():
                df = pd.DataFrame(value, columns = ['epi', 'timestep', key])
                df.to_csv(self.__create_target_filename(key))
        if self.vector_dict is not None:
            for key, value in self.vector_dict.items():
                df = pd.DataFrame(value, columns = ['epi', 'timestep', key])
                tags = df[key].apply(pd.Series)
                tags = tags.rename(columns = lambda x : key + str(x))
                full_df = pd.concat([df[df.columns[:2]], tags], axis=1)
                full_df.to_csv(self.__create_target_filename(key))

    def close(self):
        # self.writer.export_scalars_to_json("./all_scalars.json")
        # self.writer.close()   
        self.dict_to_files()
