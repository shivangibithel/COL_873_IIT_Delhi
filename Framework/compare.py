import os
import os.path as path
import numpy as np
import random
import os.path as path
import argparse
import matplotlib.pyplot as plt

class Comparator:
    '''
        Comparator class reads stats stored across multiple files.
        Has functions to plot bar graphs and line graphs, as per the labels provided by the user
        The main function takes 3 arguments -> filepaths, bar_labels and line_labels
        Get more info about input format : python compare.py --help
    '''
    def __init__(self, files):
        self.files = files
        assert len(files) > 0 and files[0] != '', 'No file/empty filename provided!!'
        self.stats = [self.load_stats(f) for f in files]
        self.labels = [path.basename(f).split('.')[0] for f in files]
        self.comparisions = {}
        self.outdir = '_V_'.join(self.labels)
        if not path.exists(self.outdir):
            os.makedirs(self.outdir)

    def load_stats(self, file):
        encoding = 'ASCII'
        if 'py2' in file:
            # This is because files dumped in python2 have different encoding
            encoding = 'latin1'
        return np.load(file, allow_pickle=True, encoding=encoding)[()]

    def createBarPlot(self, tag, subtag1='', subtag2=''):
        '''
            creates a bar plot
            supported tags - params_size, training_time, inference_time
            other tags can also be used, but consistency has to be ensured by the user
            Here upto 3 levels can be handled. For example : stats['data_stats']['train']['num_samples']
        ''' 
        labels = self.labels
        values = [s[tag] for s in self.stats]
        if subtag1 != '':
            values = [s[subtag1] for s in values]
            if subtag2 != '':
                values = [s[subtag2] for s in values]
        plt.title('{} {} {}'.format(tag, subtag1, subtag2))
        plt.bar(labels, values)
        outfile = path.join(self.outdir, '{} {} {}.jpeg'.format(tag, subtag1, subtag2))
        plt.savefig(outfile)
        plt.close()
    
    def createLinePlot(self, tag, subtag1='', subtag2='', trim=None, normalize_x=False, normalize_y=False):
        '''
            supported tags: loss_histry, metrics
            If the metric object is 2-level, then a subtag can be provided as well
            For example -> self.stats1[tag][subtag]
            Note subtag2 != '' only if subtag1 != ''
            This function does not handle 3-level tags, unlike createBarPlot()
            Here, when 2nd tag is provided then 1st tag serves as y-axis and 2nd as x-axis
            For example -> precision-recall curves
        '''
        is_biaxial = False
        subtag = subtag1
        if subtag1 != '' and subtag2 != '':
            is_biaxial = True
            subtag = subtag1 + ' ' + subtag2
        
        y_label = tag + ' ' + subtag1
        x_label = 'i' if subtag2 == '' else (tag + ' ' + subtag2)

        for i in range(len(self.labels)):
            label = self.labels[i]
            line = self.stats[i][tag]
            if subtag1 != '':
                line = line[subtag1]

            # ######## UNCOMMENT THIS SECTION IF USING THE STATS PROVIDED IN THE GIT-REPO
            # if tag=='loss_history':
            #     line = line[1:]
            # ###########
            indices = [j for j in range(1,len(line)+1)]
            # In this case we replace indices by subtag2 readings
            if subtag2 != '':
                indices = self.stats[i][tag][subtag2]

            if trim:
                indices = indices[:trim]
                line = line[:trim]
            if normalize_x:
                indices = indices/np.linalg.norm(indices)
            if normalize_y:
                line = line/np.linalg.norm(line)

            plt.plot(indices, line, label=label)       

        plt.title(tag+ ' ' + subtag)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.legend()
        outfile = path.join(self.outdir, '{}.jpeg'.format(tag+' '+subtag))
        plt.savefig(outfile)
        plt.close()


def get_arguments():
    parser = argparse.ArgumentParser()
    a = parser.add_argument
    metrics = ['pre_ttoi', 'map_ttoi', 'recall_ttoi', 'pre_itot', 'map_itot', 'recall_itot']
    metrics_attributes = ','.join(['metrics:{}'.format(m) for m in metrics])
    metrics_dual_attributes = 'metrics:pre_ttoi:recall_ttoi, metrics:pre_itot:recall_itot'
    default_line_tags = ','.join([metrics_attributes, metrics_dual_attributes, 'loss_history'])

    default_bar_tags = 'training_time, params_size, prediction_time'
    # default_bar_tags = 'training_time, params_size, prediction_time, data_stats:train:num_samples, data_stats:train:num_classes'


    a('--filepaths', type=str, 
            default='', 
            help='comma separated relative filepaths for stats files of algorithms')
    a('--bar_tags', type=str, default=default_bar_tags, help='Same as the help for line_tags')
    a('--line_tags', type=str, 
            default=default_line_tags,
            help='comma separated attributes to compare. For subtags following tags, use :\
                \n Supports only 2-level access. Value after 1st colon is treated as subtag1\
                \n For example - metrics:pre_ttoi will plot values of pre_ttoi metric\
                \n If a 2nd colon is used, then it is assigned subtag2\
                \n For example - metrics:pre_ttoi:recall_ttoi plots precision vs recall values\
                \n ,i.e. if 2 subtags are given then they are assumed to share the same parent tag and the graph is plotted as subtag1 vs subtag2\
                \n Note: for some tags, it is advisable to plot for only 1 file. Example: loss_history')

    return parser.parse_args()

# NOTE : main function and the above arguments are just to demonstrate the functionality of the Comparator class
# In general, one can write similar wrapper code to use the Comparator class, with modifications, if needed
if __name__ == '__main__':
    arguments = get_arguments()
    filepaths = arguments.filepaths
    files = filepaths.split(',')
    files = [f.strip() for f in files]

    bar_tags_raw = arguments.bar_tags
    line_tags_raw = arguments.line_tags

    bar_tags = bar_tags_raw.split(',')
    bar_tags = [t.strip() for t in bar_tags]
    bar_tags = [ [val.strip() for val in t.split(':')] for t in bar_tags]

    line_tags = line_tags_raw.split(',')
    line_tags = [t.strip() for t in line_tags]
    line_tags = [ [val.strip() for val in t.split(':')] for t in line_tags]

    comparisions = Comparator(files)
    for bt in bar_tags:
        if len(bt) == 1:
            comparisions.createBarPlot(bt[0])
        elif len(bt) == 2:
            comparisions.createBarPlot(bt[0], bt[1])
        else:
            comparisions.createBarPlot(bt[0], bt[1], bt[2])

    
    for lt in line_tags:       
        normalize_y = False
        trim = None
        # UNCOMMENT THIS FOR INTER-MODEL COMPARISION
        # if 'loss_history' == lt[0]:
        #     normalize_y = True
        # if 'metrics' == lt[0]:
        #     trim = 2000
        # UNCOMMENT THIS FOR INTER-MODEL COMPARISION
        
        if len(lt) == 1:
            comparisions.createLinePlot(lt[0], normalize_y=normalize_y, trim=trim)
        elif len(lt) == 2:
            comparisions.createLinePlot(lt[0], lt[1], trim=trim)
        else:
            comparisions.createLinePlot(lt[0], lt[1], lt[2], trim=trim)

    print (bar_tags)
    print (line_tags)