#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logparser import LogSieve, evaluator
import pandas as pd


input_dir = '../logs/' # The input directory of log file
output_dir = 'LogSieve_result/' # The output directory of parsing results

benchmark_settings = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(?:/)?(\d+\.){3}\d+(?::\d+)?', r'(\d+\.){3}\d+(:\d+)?'],
        'st': 0.5,
        'depth': 4
        },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'regex': [r'\b[\w]+(?:_[\w]+)+\b', r'(\d+\.){3}\d+', r'hdfs://[\w.-]+(?:[:\d]+)?(?:/[\w./-]*)?'],
        'st': 0.5,
        'depth': 4        
        },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
        'regex': [r'(\d+\.){3}\d+', r'\d+(\.\d+)?\s?[KMGT]B', r'([\w-]+\.){2,}[\w-]+'],
        'regex': [],
        'st': 0.5,
        'depth': 4
        },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
        'st': 0.5,
        'depth': 4        
        },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\K.*', r'\.{5,}\K.*'],
        'st': 0.5,
        'depth': 4        
        },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'regex': [r'=\d+'],
        'st': 0.5,
        'depth': 4
        },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\b[a-zA-Z]{2}\d+\b(?:\s+[a-zA-Z]{2}\d+\b)*'],
        'st': 0.6,
        'depth': 4        
        },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'regex': [r'0x.*?\s'],
        'st': 0.7,
        'depth': 5      
        },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d{1,3}\.){3}\d{1,3}(#[0-9]+)?', r'\d{2}:\d{2}:\d{2}', r'/(?:[\w\-./_]+)(?:\?[^\s"]+)?(?="|\s)'],
        'st': 0.6,
        'depth': 6        
        },

    'Android': {
        'log_file': 'Android/Android_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
        'st': 0.2,
        'depth': 6   
        },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'regex': [r'\bcom(?:\.[a-zA-Z0-9_]+)+(?:[@/][a-zA-Z0-9_.]+)?\b', r'\d+(?:\s+\d+){3}', r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', r'/(?:[\w\-.:/_]+)(?:\?[^\s"]+)?(?="|\s|\)|$)', r'\[[\d,\s]*\]'],
        'st': 0.2,
        'depth': 4
        },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'regex': [r'(\d+\.){3}\d+', r'(?:HTTP?)?/(?:[\w\-./_]+)(?:\?[^\s"]+)?(?="|\s|\)|$)', r'\b(?:vm:|[\w.-]+:[\w.-]+)(?=\W|$)'],
        'st': 0.5,
        'depth': 4        
        },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'regex': [r'<\d+\ssec', r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(:\d+)?\b', r'\d{2}:\d{2}(:\d{2})*', r'\d+(\.\d+)?\s?[KMGT]B', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\b[a-zA-Z0-9_-]+:\d+\b'],
        'st': 0.88,
        'depth': 3
        },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'regex': [r"'[^']+'",  r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
        'st': 0.6,
        'depth': 5   
        },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'regex': [r'((\d+\.){3}\d+,?)+', r'HTTP/1\.1', r'/(?:[\w\-./_]+)(?:\?[^\s"]+)?(?="|\s|\)|$)', r'\b[a-zA-Z0-9]+(?:[-.][a-zA-Z0-9]+)+\b'],
        'st': 0.7,
        'depth': 5
        },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'regex': [r'([\w-]+\.){2,}[\w-]+', r'\b(en0)\b'],
        'st': 0.7,
        'depth': 6   
        },
}


def main():
    bechmark_result = []
    for dataset, setting in benchmark_settings.iteritems():
        print('\n=== Evaluation on %s ==='%dataset)
        indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
        log_file = os.path.basename(setting['log_file'])

        parser = LogSieve.LogParser(log_format=setting['log_format'], indir=indir, outdir=output_dir, rex=setting['regex'], depth=setting['depth'], st=setting['st'] )
        parser.parse(log_file)

        F1_measure, accuracy = evaluator.evaluate(
                               groundtruth=os.path.join(indir, log_file + '_structured.csv'),
                               parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
                               )
        bechmark_result.append([dataset, F1_measure, accuracy])

    print('\n=== Overall evaluation results ===')
    df_result = pd.DataFrame(bechmark_result, columns=['Dataset', 'F1_measure', 'Accuracy'])
    df_result.set_index('Dataset', inplace=True)
    print(df_result)
    df_result.T.to_csv('LogSieve_bechmark_result.csv')


if __name__ == '__main__':
    main()
