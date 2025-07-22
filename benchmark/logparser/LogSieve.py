"""
Description : This file implements the LogSieve algorithm for log parsing
Author      : LogPAI team
License     : MIT
"""
import random
import string

import regex as re
import os
import numpy as np
import pandas as pd
import hashlib
from datetime import datetime
from textblob import TextBlob, Word
from nltk.stem import WordNetLemmatizer
from functools import lru_cache  # 导入缓存装饰器
from datasketch import MinHashLSH, MinHash
import uuid
import json
from openai import OpenAI
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

class Logcluster:
   def __init__(self, logTemplate='', logIDL=None):
       self.logTemplate = logTemplate
       self.logIDL = logIDL if logIDL is not None else []
       # 根据模板内容生成唯一uuid
       self.uuid = hashlib.md5(' '.join(logTemplate).encode()).hexdigest()[:8]
       # 新增LLM解析标记（默认未解析）
       self.parsed_by_llm = False





class Node:
    def __init__(self, childD=None, depth=0, digitOrtoken=None):
        if childD is None:
            childD = dict()
        self.childD = childD
        self.depth = depth
        self.digitOrtoken = digitOrtoken


class LogParser:

    # 在类中添加以下特征权重配置
    FEATURE_WEIGHTS = {
        'wildcard_ratio': 0.1,        # 通配符比例
        'consecutive_wildcards': 0.15,  # 降低连续权重
        'wildcard_density': 0.3,       # 新增密度波动特征
        'symbol_density': 0.25,        # 符号密度
        'numeric_density': 0.2,         # 数字密度
    }
    ANOMALY_THRESHOLD = 0.65          # 动态调整基准值

    def __init__(self, log_format, indir='./', outdir='./result/', depth=4, st=0.4,
                 maxChild=100, rex=[], keep_para=True ):
        """
        Attributes
        ----------
            rex : regular expressions used in preprocessing (step1)
            path : the input path stores the input log file name
            depth : depth of all leaf nodes
            st : similarity threshold
            maxChild : max number of children of an internal node
            logName : the name of the input file containing raw log messages
            savePath : the output path stores the file containing structured logs
        """
        self.path = indir
        self.depth = depth - 2
        self.st = st
        self.maxChild = maxChild
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex
        self.keep_para = keep_para
        self.common_set = self.load_common_words()
        self.lemmatizer = WordNetLemmatizer()
        self.similar_clusters = []
        # 预编译正则表达式提升效率
        self.placeholder_re = re.compile(r'<.*?>')
        self.alpha_re = re.compile(r'^[A-Za-z]+$')
        self.non_word_char_re = re.compile(r'[^\w\s]')

        # 新增预编译正则表达式
        self.special_symbols = re.compile(r'^[^\w\s(){}\[\]=,]$')  # 新增排除=和,
        self.bracket_symbols = re.compile(r'[(){}\[\]<>=,]')  # 新增包含=和,

        self.template_metrics = defaultdict(dict)  # 存储模板指标

    def hasNumbers(self, s):
        return any(char.isdigit() for char in s)

    def treeSearch(self, rn, seq):
        retLogClust = None
        seqLen = len(seq)

        # 检查当前长度的日志簇
        if seqLen in rn.childD:
            parentn = rn.childD[seqLen]
            retLogClust = self.searchInNode(parentn, seq)
            if retLogClust is not None:
                return retLogClust

        """
        # 检查seq中是否包含特定token
        if '(' in seq or ':' in seq:
            # 检查比当前长度短3到5的日志簇
            for offset in range(2, 6):
                if seqLen - offset in rn.childD:
                    parentn = rn.childD[seqLen - offset]
                    retLogClust = self.searchInNode(parentn, seq)
                    if retLogClust is not None:
                        return retLogClust
        """

        return retLogClust

    def searchInNode(self, parentn, seq):
        currentDepth = 1
        for token in seq:
            if currentDepth >= self.depth or currentDepth > len(seq):
                break

            if token in parentn.childD:
                parentn = parentn.childD[token]
            elif '<*>' in parentn.childD:
                parentn = parentn.childD['<*>']
            else:
                return None
            currentDepth += 1

        logClustL = parentn.childD
        return self.fastMatch(logClustL, seq)

    def addSeqToPrefixTree(self, rn, logClust):
        seqLen = len(logClust.logTemplate)
        if seqLen not in rn.childD:
            firtLayerNode = Node(depth=1, digitOrtoken=seqLen)
            rn.childD[seqLen] = firtLayerNode
        else:
            firtLayerNode = rn.childD[seqLen]

        parentn = firtLayerNode

        currentDepth = 1
        for token in logClust.logTemplate:

            #Add current log cluster to the leaf node
            if currentDepth >= self.depth or currentDepth > seqLen:
                if len(parentn.childD) == 0:
                    parentn.childD = [logClust]
                else:
                    parentn.childD.append(logClust)
                break

            #If token not matched in this layer of existing tree.
            if token not in parentn.childD:
                if not self.hasNumbers(token):
                    if '<*>' in parentn.childD:
                        if len(parentn.childD) < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD['<*>']
                    else:
                        if len(parentn.childD)+1 < self.maxChild:
                            newNode = Node(depth=currentDepth+1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        elif len(parentn.childD)+1 == self.maxChild:
                            newNode = Node(depth=currentDepth+1, digitOrtoken='<*>')
                            parentn.childD['<*>'] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD['<*>']

                else:
                    if '<*>' not in parentn.childD:
                        newNode = Node(depth=currentDepth+1, digitOrtoken='<*>')
                        parentn.childD['<*>'] = newNode
                        parentn = newNode
                    else:
                        parentn = parentn.childD['<*>']

            #If the token is matched
            else:
                parentn = parentn.childD[token]

            currentDepth += 1

    #seq1 is template
    def load_common_words(self):

        # 获取当前脚本所在目录
        current_dir = os.path.dirname(__file__)
        filename = os.path.join(current_dir, "oxford-5000.csv")

        print("Oxford-5000.csv当前工作目录:", filename)

        """
        从oxford-5000.csv文件的word列中加载常用单词列表（转换为小写）
        """
        common_set = set()
        df = pd.read_csv(filename, encoding='utf-8')
        for word in df['word']:
            if pd.notna(word):  # 确保单词不为空
                common_set.add(word.lower())
        return common_set

    @lru_cache(maxsize=2048)  # 增大缓存容量
    def is_common_word(self, token):
        """优化后的常见词判断"""
        # 过滤占位符
        if self.placeholder_re.fullmatch(token):
            return False, token

        # 仅处理纯字母单词
        if not self.alpha_re.match(token):
            return False, token

        lower_token = token.lower()
        # 直接检查小写形式
        if lower_token in self.common_set:
            return True, lower_token

        # 尝试不同词性的词形还原（按常见顺序）
        lemma = self.lemmatizer.lemmatize(lower_token)  # 默认名词
        if lemma in self.common_set:
            return True, lemma

        # 尝试其他词性：动词、形容词、副词
        for pos in ['v', 'a', 'r']:
            current_lemma = self.lemmatizer.lemmatize(lower_token, pos=pos)
            if current_lemma in self.common_set:
                return True, current_lemma

        return False, lower_token

    def seqDist(self, seq1, seq2):
        """优化后的相似度计算"""
        assert len(seq1) == len(seq2)

        sim_weight = 0.0
        total_weight = 0.0
        numOfPar = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == '<*>':
                numOfPar += 1
                continue

            # 初始化lemma变量避免未定义
            lemma1 = lemma2 = None

            # 基础权重判断（使用预编译正则）
            base1 = 1.0 if self.alpha_re.fullmatch(token1) else 0.25
            base2 = 1.0 if self.alpha_re.fullmatch(token2) else 0.25

            if self.alpha_re.fullmatch(token1) and self.alpha_re.fullmatch(token2):
                is_common1, lemma1 = self.is_common_word(token1)
                is_common2, lemma2 = self.is_common_word(token2)

                if lemma1 == lemma2:
                    weight = max(base1, base2)
                else:
                    factor1 = 4.0 if is_common1 else 0.5
                    factor2 = 4.0 if is_common2 else 0.5
                    weight = max(base1 * factor1, base2 * factor2)
            else:
                # 处理非字母token的情况
                lemma1 = token1.lower() if self.alpha_re.fullmatch(token1) else token1
                lemma2 = token2.lower() if self.alpha_re.fullmatch(token2) else token2

                # 特殊符号处理
                if self.non_word_char_re.fullmatch(token1) and token1 != token2:
                    base1 = 4.0
                if self.non_word_char_re.fullmatch(token2) and token1 != token2:
                    base2 = 4.0
                weight = max(base1, base2)

            total_weight += weight
            if lemma1 == lemma2:
                sim_weight += weight

        return (sim_weight / total_weight, numOfPar) if total_weight > 0 else (0.0, numOfPar)


    def fastMatch(self, logClustL, seq):
        retLogClust = None

        maxSim = -1
        maxNumOfPara = -1
        maxClust = None

        for logClust in logClustL:
            curSim, curNumOfPara = self.seqDist(logClust.logTemplate, seq)
            if curSim>maxSim or (curSim==maxSim and curNumOfPara>maxNumOfPara):
                maxSim = curSim
                maxNumOfPara = curNumOfPara
                maxClust = logClust

        if maxSim >= self.st:
            retLogClust = maxClust

        return retLogClust

    def getTemplate(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        retVal = []

        i = 0
        for word in seq1:
            if word == seq2[i]:
                retVal.append(word)
            else:
                retVal.append('<*>')

            i += 1

        return retVal

    def generate_cluster_structured_data(self, clusters, max_samples=3):
        """
        生成聚类结构化数据（可复用方法）
        :param clusters: 聚类分组列表，每个元素是Logcluster对象列表
        :param max_samples: 每个模板最大示例日志数
        :return: 结构化数据列表
        """
        structured_data = []
        for cluster_id, cluster_group in enumerate(clusters, 1):
            templates_info = []
            total_logs = 0

            # 收集模板信息
            for cluster in cluster_group:
                # 获取示例日志
                sample_logs = self.get_sample_logs(cluster.logIDL, max_samples=max_samples)
                # 生成模板字符串
                template_str = ' '.join(cluster.logTemplate)

                templates_info.append({
                    'template': template_str,
                    'samples': sample_logs,
                    'template_id': cluster.uuid  # 携带模板唯一标识
                })
                total_logs += len(cluster.logIDL)

            # 构建集群条目（移除anomaly_score字段）
            cluster_entry = {
                "cluster_id": f"C{cluster_id:04d}",
                "templates": templates_info,
                "total_occurrences": total_logs
            }
            structured_data.append(cluster_entry)

        return structured_data

    def save_cluster_results(self, clusters, format='json'):
        """
        持久化存储聚类结果（改进版）
        :param clusters: 聚类结果列表，每个元素是Logcluster对象列表
        :param format: 存储格式，支持json/csv
        """

        print(f"[Debug] 收到聚类组数量: {len(clusters)}")
        for i, group in enumerate(clusters, 1):
            print(f"组{i}包含模板数: {len(group)}")
            for j, cluster in enumerate(group, 1):
                print(f"  模板{j}: {cluster.logTemplate}")

        if not clusters:
            print("Warning: Empty cluster results, skip saving")
            return

        # 生成结构化数据
        structured_data = self.generate_cluster_structured_data(clusters)

        # 创建保存路径
        save_dir = os.path.join(self.savePath, "clusters")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.logName}_clusters.{format}")

        try:
            if format == 'json':
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(structured_data, f, indent=2, ensure_ascii=False)
            elif format == 'csv':
                # 将数据转换为适合CSV的格式
                csv_data = []
                for entry in structured_data:
                    cluster_id = entry['cluster_id']
                    for template in entry['templates']:
                        csv_data.append({
                            "ClusterID": cluster_id,
                            "Template": template['template'],
                            "TemplateID": template['template_id'],
                            "Samples": " | ".join(template['samples']),
                            "TotalOccurrences": entry['total_occurrences']
                        })
                pd.DataFrame(csv_data).to_csv(save_path, index=False)
            print(f"聚类结果已保存至: {save_path}")
        except Exception as e:
            print(f"保存聚类结果失败: {str(e)}")

    def clean_sentence_to_tokens(self, sentence: str) -> List[str]:
        punctuation = string.punctuation
        raw_tokens = sentence.split()
        cleaned_tokens = []
        for token in raw_tokens:
            while len(token) > 0 and token[0] in punctuation:
                token = token[1:]
            while len(token) > 0 and token[-1] in punctuation:
                token = token[:-1]
            if token:
                cleaned_tokens.append(token)
        return cleaned_tokens

    def compute_token_complexity(self, token: str) -> int:
        segments = re.findall(r'[A-Za-z]+|[0-9]+|[^A-Za-z0-9]', token)
        return len(segments)

    def extract_entity_list_from_df_log(self, window_size=50, tc_threshold=3, rf_threshold=2, neighbor_count=20, debug=False):
        # 预处理：为每条日志提取高复杂度 token
        high_tc_token_list = []
        for content in self.df_log['Content']:
            tokens = self.clean_sentence_to_tokens(content)
            high_tc_tokens = [
                token for token in tokens
                if self.compute_token_complexity(token) >= tc_threshold
            ]
            high_tc_token_list.append(high_tc_tokens)

        self.df_log['HighTC'] = high_tc_token_list

        # 构建 eventId → 行号 的索引字典
        eventid_to_indices = defaultdict(list)
        for idx, eid in enumerate(self.df_log['EventId']):
            eventid_to_indices[eid].append(idx)

        # 为每条日志计算RF并提取最终实体
        final_entities = []
        for i in range(len(self.df_log)):
            current_eventid = self.df_log.at[i, 'EventId']
            context_indices = set()

            # 滑动窗口（包含前后日志）
            start = max(0, i - window_size)
            end = min(len(self.df_log), i + window_size + 1)
            context_indices.update(range(start, end))

            # 添加相邻"不同模板"的日志
            up, down = i - 1, i + 1
            found_up = found_down = 0
            while (up >= 0 or down < len(self.df_log)) and (found_up < neighbor_count or found_down < neighbor_count):
                if up >= 0 and self.df_log.at[up, 'EventId'] != current_eventid:
                    context_indices.add(up)
                    found_up += 1
                if down < len(self.df_log) and self.df_log.at[down, 'EventId'] != current_eventid:
                    context_indices.add(down)
                    found_down += 1
                up -= 1
                down += 1

            # 收集上下文中的所有高复杂度token
            context_tokens = []
            for idx in context_indices:
                context_tokens.extend(self.df_log.at[idx, 'HighTC'])

            rf_counter = Counter(context_tokens)

            # 当前日志中哪些token满足 RF ≥ 阈值
            current_tokens = self.df_log.at[i, 'HighTC']
            entity_tokens = list({tok for tok in current_tokens if rf_counter[tok] >= rf_threshold})

            if debug:
                print(f"\nLineId: {self.df_log.at[i, 'LineId']} | Context lines: {len(context_indices)}")
                print(f"Current Tokens: {current_tokens}")
                print(f"RF Counter: {dict(rf_counter)}")
                print(f"Final Entity Tokens: {entity_tokens}")

            final_entities.append(entity_tokens)

        # 添加结果列并删除中间列
        self.df_log['Entity'] = final_entities
        self.df_log.drop(columns=['HighTC'], inplace=True)

    def outputResult(self, tokens_with_context_list, logClustL):

        log_templates = [''] * self.df_log.shape[0]
        log_templateids = [''] * self.df_log.shape[0]
        df_events = []

        for logClust in logClustL:
            # 确保模板内容都是字符串
            cleaned_template = [str(token) for token in logClust.logTemplate]
            if logClust.parsed_by_llm:
                template_str = ' '.join(cleaned_template)
            else:
                template_str = self.join_tokens_with_context(
                    tokens_with_context_list[logClust.logIDL[0] - 1],
                    cleaned_template  # 使用清理后的模板
                )
            occurrence = len(logClust.logIDL)
            template_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for logID in logClust.logIDL:
                logID -= 1
                log_templates[logID] = template_str
                log_templateids[logID] = template_id
            df_events.append([template_id, template_str, occurrence])

        df_event = pd.DataFrame(df_events, columns=['EventId', 'EventTemplate', 'Occurrences'])
        self.df_log['EventId'] = log_templateids
        self.df_log['EventTemplate'] = log_templates
        """
        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)
        """

        # 新增实体提取步骤
        if not hasattr(self.df_log, 'Entity'):  # 避免重复处理
            self.extract_entity_list_from_df_log(
                window_size=50,
                tc_threshold=3,
                rf_threshold=2,
                neighbor_count=20
            )

        self.df_log.to_csv(os.path.join(self.savePath, self.logName + '_structured.csv'), index=False)


        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['EventId'] = df_event['EventTemplate'].map(lambda x: hashlib.md5(str(x).encode('utf-8')).hexdigest()[0:8])
        df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)
        df_event.to_csv(os.path.join(self.savePath, self.logName + '_templates.csv'), index=False, columns=["EventId", "EventTemplate", "Occurrences"])

    def merge_wildcards(self, tokens):
        merged = []
        prev_is_wildcard = False
        for token in tokens:
            if token == '<*>':
                if not prev_is_wildcard:
                    merged.append(token)
                    prev_is_wildcard = True
            else:
                merged.append(token)
                prev_is_wildcard = False
        return merged

    def find_similar_templates(self, logClustL, sim_threshold=0.85):

        """
        改进版：直接从logClustL收集所有模板，避免遍历LogSieve树
        """
        # 步骤1：直接从logClustL收集所有模板
        # 根据uuid去重
        unique_clusters = {cluster.uuid: cluster for cluster in logClustL}.values()
        all_clusters = list(unique_clusters)

        print(f"========== 模板收集 ==========")
        print(f"[Direct Collection] 直接从{len(logClustL)}个日志集群收集到{len(all_clusters)}个模板")
        print(f"========== 模板收集 ==========")

        # 使用MinHash LSH进行快速相似性搜索
        lsh = MinHashLSH(threshold=sim_threshold, num_perm=128)
        minhashes = {}
        for cluster in all_clusters:
            mh = MinHash(num_perm=128)
            processed_tokens = self.merge_wildcards(cluster.logTemplate)
            for token in processed_tokens:
                mh.update(token.encode('utf-8'))
            lsh.insert(cluster.uuid, mh)
            minhashes[cluster.uuid] = mh

        # 构建相似性图
        similarity_graph = defaultdict(set)
        for cluster in all_clusters:
            results = lsh.query(minhashes[cluster.uuid])
            for matched_uuid in results:
                if matched_uuid != cluster.uuid:
                    similarity_graph[cluster.uuid].add(matched_uuid)

        # 并查集进行聚类
        parent = {uuid: uuid for uuid in minhashes}

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def union(u, v):
            pu, pv = find(u), find(v)
            if pu != pv:
                parent[pv] = pu

        for u in similarity_graph:
            for v in similarity_graph[u]:
                union(u, v)

        clusters = defaultdict(list)
        for uuid in parent:
            clusters[find(uuid)].append(uuid)

        # 将uuid转换为对应的Logcluster对象
        id_to_cluster = {cluster.uuid: cluster for cluster in all_clusters}
        clustered_groups = [
            [id_to_cluster[uuid] for uuid in group]
            for group in clusters.values()
            if len(group) >= 2  # 仅保留包含至少两个模板的聚类
        ]

        print(f"[Debug] 最终聚类分组结构: {type(clustered_groups)}")
        print(f"第一组类型: {type(clustered_groups[0]) if clustered_groups else '无数据'}")
        print(f"第一组元素类型: {type(clustered_groups[0][0]) if clustered_groups else '无数据'}")

        return clustered_groups

    def printTree(self, node, dep):
        pStr = ''
        for i in range(dep):
            pStr += '\t'

        if node.depth == 0:
            pStr += 'Root'
        elif node.depth == 1:
            pStr += '<' + str(node.digitOrtoken) + '>'
        else:
            pStr += node.digitOrtoken

        print(pStr)

        if node.depth == self.depth:
            return 1
        for child in node.childD:
            self.printTree(node.childD[child], dep + 1)

    def parse(self, logName):

        # 获取当前工作目录
        current_directory = os.getcwd()

        # 打印当前工作目录
        print("当前工作目录:", current_directory)

        print('Parsing file: ' + os.path.join(self.path, logName))
        start_time = datetime.now()
        self.logName = logName
        rootNode = Node()
        logCluL = []
        tokens_with_context_list = []  # ´æ´¢Ã¿Ò»ÐÐÈÕÖ¾µÄ tokens_with_context

        self.load_data()

        count = 0
        for idx, line in self.df_log.iterrows():
            logID = line['LineId']
            tokens_with_context, logmessageL = self.preprocess(line['Content'])
            tokens_with_context_list.append(tokens_with_context)  # ´æ´¢µ±Ç°ÐÐµÄ tokens_with_context
            logmessageL = logmessageL.strip().split()
            logmessageL = self.replace_integers_with_placeholder(logmessageL)
            matchCluster = self.treeSearch(rootNode, logmessageL)

            # Match no existing log cluster
            if matchCluster is None:
                newCluster = Logcluster(logTemplate=logmessageL, logIDL=[logID])
                logCluL.append(newCluster)
                self.addSeqToPrefixTree(rootNode, newCluster)

            # Add the new log message to the existing cluster
            else:
                newTemplate = self.getTemplate(logmessageL, matchCluster.logTemplate)
                matchCluster.logIDL.append(logID)
                if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate):
                    matchCluster.logTemplate = newTemplate

            count += 1
            if count % 100000 == 0 or count == len(self.df_log):
                print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))


        start_cluster = datetime.now()
        # 在parse方法中调用

        # ====== 新增统计代码 ======
        print("\n" + "=" * 40)
        print(f"=== 初始日志簇数量: {len(logCluL)} ===".center(30))
        print("=" * 40 + "\n")
        # ====== 新增结束 ======

        unique_clusters = []
        seen = set()
        for cluster in logCluL:
           key = ' '.join(cluster.logTemplate)
           if key not in seen:
               seen.add(key)
               unique_clusters.append(cluster)

        # 修改后应变为
        template_cluster_groups = self.find_similar_templates(unique_clusters, sim_threshold=0.8)


        # 保存相似聚类信息到实例变量
        # self.similar_clusters = template_clusters
        self.similar_clusters = template_cluster_groups  # 现在存储的是Logcluster对象的分组



        """
        for idx, cluster in enumerate(template_clusters):
            print(f"Cluster {idx + 1}:")
            for tpl in cluster:
                print("  ", ' '.join(tpl))
        """

        # 生成结构化数据
        structured_data = self.generate_cluster_structured_data(template_cluster_groups)

        self.save_cluster_results(
            clusters=template_cluster_groups,
            format='json'  # 可选json/csv
        )

        cluster_time = datetime.now() - start_cluster
        print(f"\n模板聚类完成，耗时: {cluster_time}")
        print(f"发现 {len(template_cluster_groups)} 个有效聚类组")



        print("\n=== 开始相似大模型优化 ===")
        refiner = ClusterRefiner()
        optimized_clusters = refiner.refine_clusters(logCluL, structured_data)

        # 更新原始集群数据
        logCluL = optimized_clusters




        # ====== 新增统计代码 ======
        print("\n" + "=" * 40)
        print(f"=== 聚类更新后-异常检测前日志簇数量: {len(logCluL)} ===".center(30))
        print("=" * 40 + "\n")
        # ====== 新增结束 ======



        # 异常检测与保存
        anomalous_templates = self.detect_anomalous_templates(logCluL)
        self.save_anomalous_templates(anomalous_templates)


        anomaly_refiner = AnomalyRefiner()
        optimized_clusters = anomaly_refiner.refine_anomalies(logCluL, anomalous_templates)
        logCluL = optimized_clusters



        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        self.outputResult(tokens_with_context_list, logCluL)  # ´«µÝ tokens_with_context_list

        # print("\nPrinting the Prefix Tree:")
        # self.printTree(rootNode, 0)

        # ====== 新增统计代码 ======
        print("\n" + "=" * 40)
        print(f"=== 异常检测后日志簇数量: {len(logCluL)} ===".center(30))
        print("=" * 40 + "\n")
        # ====== 新增结束 ======

        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)

    def preprocess(self, line):

        # 去除行开头的空格
        line = line.lstrip()

        # 1. ¸ù¾ÝÕýÔòÁÐ±íÌæ»»Æ¥Åä²¿·ÖÎª '<*>'
        for currentRex in self.rex:
            line = re.sub(currentRex, '<*>', line)

        # 2. ÔÚÎªÌØ¶¨·ûºÅÌí¼Ó¿Õ¸ñÖ®Ç°£¬²¶»ñÃ¿¸ö token µÄÇ°ÖÃºÍºóÖÃ¿Õ°×ÐÅÏ¢
        # (\s*) Æ¥ÅäÇ°ÖÃ¿Õ°×£»(\S+) Æ¥Åä·Ç¿Õ°××Ö·û£»(\s*) Æ¥ÅäºóÖÃ¿Õ°×
        pattern = re.compile(r'(\s*)((?:[()\[\]=:,])|(?:[^\s()\[\]=:,]+))(\s*)')
        tokens_with_context = [match.groups() for match in pattern.finditer(line)]

        # 3. ÎªÌØ¶¨·ûºÅÌí¼Ó¿Õ¸ñ£¬Ê¹µÃÕâÐ©·ûºÅ³ÉÎª¶ÀÁ¢µÄ token
        processed_line = re.sub(r'([()\[\]=:,])', r' \1 ', line)

        return tokens_with_context, processed_line

    def join_tokens_with_context(self, token_contexts, processed_tokens):
        """
        ÀûÓÃ¼ÇÂ¼µÄÃ¿¸ö token µÄÇ°ÖÃºÍºóÖÃ¿Õ¸ñÐÅÏ¢£¬½«´¦ÀíºóµÄ token °´ÕÕÔ­Ê¼Ë³ÐòÆ´½ÓÆðÀ´¡£
        """
        if len(token_contexts) != len(processed_tokens):
            print(f"Error processing content: {token_contexts} + {processed_tokens}")
            # raise ValueError("Length mismatch between token contexts and processed tokens")

        result = []
        for (leading, original_token, trailing), token in zip(token_contexts, processed_tokens):
            result.append(leading + token + trailing)
        return ''.join(result)

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        print("Total lines: ", len(logdf))
        return logdf

    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def replace_integers_with_placeholder(self, logmessageL):
        """
        Replace integer tokens (including negative integers) with <*> in the log message list.

        Parameters:
        logmessageL (list): List of tokens from the log message.

        Returns:
        list: Processed list of tokens with integers replaced by <*>.
        """
        processed_logmessageL = []
        for token in logmessageL:
            if re.match(r'^(?=.*\d)[\d+\-.*%&$#/]+$', token):  # Match integers including negative integers
                processed_logmessageL.append('<*>')
            else:
                processed_logmessageL.append(token)
        return processed_logmessageL

    def get_parameter_list(self, row):
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex:
            return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r'\\ +', r'\\s+', template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return parameter_list

    def calculate_template_metrics(self, template):
        """综合特征计算引擎"""
        metrics = {}
        tokens = template

        # 1. 通配符特征（比例 + 分布）
        total_tokens = len(tokens)
        wildcard_count = tokens.count('<*>')
        metrics['wildcard_ratio'] = wildcard_count / (total_tokens * 1.5) if total_tokens > 0 else 0

        # 2. 连续通配符特征（最长连续长度 + 出现次数）
        max_consecutive, consecutive_counts = self._calc_consecutive_wildcards(tokens)
        metrics['consecutive_wildcards'] = (
            0.4 * min(max_consecutive/5, 1) +
            0.6 * min(consecutive_counts/3, 1)
        )

        # 3. 新增通配符密度波动检测
        metrics.update(self._calc_wildcard_density(tokens))

        # 4. 符号密度（非字母数字任意位置出现 ≥2 次）
        metrics['symbol_density'] = max(
            (self.token_anomaly_score(token) for token in tokens if token != '<*>'),
            default=0.0  # 空值时返回0.0
        )

        # 5. 数字密度（非通配符中的数字比例）
        numeric_tokens = sum(
            1 for token in tokens
            if any(c.isdigit() for c in token) and token != '<*>'
        )
        metrics['numeric_density'] = min(numeric_tokens/3, 1)

        return metrics

    def _calc_consecutive_wildcards(self, tokens):
        """连续通配符计算（原逻辑封装）"""
        max_consecutive = current_streak = consecutive_counts = 0
        for token in tokens:
            if token == '<*>':
                current_streak += 1
                if current_streak >= 2:
                    consecutive_counts += 1
                max_consecutive = max(max_consecutive, current_streak)
            else:
                current_streak = 0
        return max_consecutive, consecutive_counts

    def _calc_wildcard_density(self, tokens, window_size=5):
        """改进版密度分析：增强模式识别能力"""
        density_metrics = {
            'wildcard_density': 0,
            'density_fluctuation': 0
        }

        if len(tokens) <= 5:
            return density_metrics

        # 动态窗口调整（新增窗口权重系数）
        original_window_size = window_size
        if 6 <= len(tokens) <= 10:
            window_size = 3
        effective_window = min(len(tokens), window_size)

        # 窗口大小权重系数（新增）
        window_weight = 1.0
        if effective_window == 3:
            window_weight = 0.8  # 降低小窗口权重

        effective_counts = []

        for i in range(len(tokens) - effective_window + 1):
            window = tokens[i:i + effective_window]
            count = 0
            pattern_weight = 1.0

            # 检测交替模式类型
            pattern_type = self._detect_alternating_pattern(window)

            # 根据模式类型调整权重（新增模式判断条件）
            if pattern_type == 'symbol_wildcard':
                pattern_weight = 1.5
                # 仅在符号模式计算符号权重（新增）
                for token in window:
                    if token == '<*>':
                        count += 1 * pattern_weight
                    elif self.special_symbols.match(token):
                        if not self.bracket_symbols.search(token):
                            count += 0.8 * pattern_weight
            elif pattern_type == 'alpha_wildcard':
                pattern_weight = 0.6
                # 字母模式只计算通配符（新增）
                count += window.count('<*>') * pattern_weight
            else:
                # 默认模式只计算通配符（新增）
                count += window.count('<*>') * pattern_weight

            # 应用窗口权重系数（新增）
            count *= window_weight
            effective_counts.append(count / effective_window)

        if not effective_counts:
            return density_metrics

        max_density = max(effective_counts)
        avg_fluctuation = np.mean(np.abs(np.diff(effective_counts))) if len(effective_counts) > 1 else 0

        # 动态权重分配（调整系数）
        density_score = (
                0.8 * max_density +  # 降低峰值影响
                0.2 * avg_fluctuation
        )

        return {
            'wildcard_density': min(density_score, 1.0),
            'density_fluctuation': avg_fluctuation
        }

    def _detect_alternating_pattern(self, window):
        """增强模式检测：更灵活的交替模式识别"""
        symbol_wildcard_count = 0
        alpha_wildcard_flag = False
        window_len = len(window)

        # 符号与通配符密集模式检测
        for token in window:
            if token == '<*>' or self.special_symbols.match(token):
                symbol_wildcard_count += 1
        if symbol_wildcard_count >= 4 or symbol_wildcard_count == window_len:
            return 'symbol_wildcard'

        # 字母与通配符交替模式检测
        if window_len >= 3:
            # 使用正则匹配类似 A-*-A 或 *-A-* 模式
            pattern_str = '-'.join(['A' if self.alpha_re.match(t) else ('W' if t == '<*>' else 'O') for t in window])
            if re.search(r'(A-W-A|W-A-W)', pattern_str):
                return 'alpha_wildcard'

        return None

    def token_anomaly_score(self, token):
        if token == '<*>':
            return 0.0

        length = len(token)
        if length < 15:
            return 0.0

        symbol_chars = set(string.punctuation)
        symbol_count = sum(1 for c in token if c in symbol_chars)
        unique_symbol_count = len(set(c for c in token if c in symbol_chars))
        uppercase_count = sum(1 for c in token if c.isupper())
        digit_count = sum(1 for c in token if c.isdigit())

        symbol_density = symbol_count / length
        uppercase_density = uppercase_count / length
        digit_density = digit_count / length

        # 特别异常的判别机制
        hard_anomaly = (
                symbol_count >= 4 or
                unique_symbol_count >= 3 or
                uppercase_count >= 5 or
                ('/' in token and length >= 20) or
                (symbol_density > 0.3 and uppercase_density > 0.2)
        )

        # 得分机制（调参版）
        score = 0
        score += 0.4 * min(symbol_density * 2, 1)  # 符号密度权重提升
        score += 0.3 * min(uppercase_density * 2, 1)  # 大写权重提升
        score += 0.2 * min(unique_symbol_count / 5, 1)  # 符号种类
        score += 0.1 * min((length - 10) / 30, 1)  # 长度加分（超长才影响）

        # 如果是极端结构，则加权放大异常评分
        if hard_anomaly:
            score = min(score * 1.5 + 0.2, 1.0)

        return round(score, 2)

    def _generate_sliding_windows(self, tokens, window_size):
        """生成滑动窗口"""
        return [tokens[i:i + window_size]
                for i in range(len(tokens) - window_size + 1)]

    def compute_anomaly_score(self, metrics):
        """加权异常评分计算"""
        return sum(
            weight * metrics.get(feature, 0)
            for feature, weight in self.FEATURE_WEIGHTS.items()
        )

    def dynamic_threshold_adjustment(self, scores):
        if not scores:
            return max(0.2, self.ANOMALY_THRESHOLD)  # 新增最低阈值

        avg = np.mean(scores)
        std = np.std(scores) if len(scores) > 1 else 0

        # 动态调整公式改进
        adjusted = avg + 2.0 * std  # 增大标准差系数
        return min(
            max(0.18, adjusted),  # 新增双保险：下限0.18，上限ANOMALY_THRESHOLD
            self.ANOMALY_THRESHOLD
        )

    def detect_anomalous_templates(self, logCluL):
        # 统计初始化
        clustered_log_count = 0  # 已聚类日志数量
        anomalous_log_count = 0  # 异常日志数量
        total_logs = self.df_log.shape[0]  # 总日志数量

        # 构建已聚类模板集合（适配Logcluster对象）
        clustered_templates = set()
        for cluster_group in self.similar_clusters:
            for log_cluster in cluster_group:
                clustered_templates.add(' '.join(log_cluster.logTemplate))
                print(' '.join(log_cluster.logTemplate))
                clustered_log_count += len(log_cluster.logIDL)  # 累加已聚类日志数

        # 第一阶段：计算所有模板指标
        all_scores = []
        template_metrics = {}

        for cluster in logCluL:
            template = cluster.logTemplate
            template_str = ' '.join(template)
            if template_str in clustered_templates:
                continue  # 跳过已聚类模板

            metrics = self.calculate_template_metrics(template)
            score = self.compute_anomaly_score(metrics)
            template_metrics[template_str] = (metrics, score)
            all_scores.append(score)

        # 动态调整阈值（保持不变）
        dynamic_threshold = self.dynamic_threshold_adjustment(all_scores)

        # 第二阶段：筛选异常模板
        anomalies = []
        for cluster in logCluL:
            template_str = ' '.join(cluster.logTemplate)
            if template_str not in template_metrics:
                continue

            metrics, score = template_metrics[template_str]
            if score >= dynamic_threshold:
                anomalous_log_count += len(cluster.logIDL)  # 累加异常日志数
                sample_logs = self.get_sample_logs(cluster.logIDL)
                anomalies.append({
                    'template': template_str,
                    'score': round(score, 2),
                    'metrics': metrics,
                    'samples': sample_logs
                })

        # 打印醒目标识
        print("\n" + "=" * 60)
        print("=== 异常检测统计 ===".center(50))
        print("=" * 60)
        print(f"总日志数量: {total_logs}")
        print(f"├─ 已聚类日志: {clustered_log_count} ({clustered_log_count / total_logs:.1%})")
        print(f"└─ 异常日志  : {anomalous_log_count} ({anomalous_log_count / total_logs:.1%})")
        total_covered = clustered_log_count + anomalous_log_count
        print(f"└─ 覆盖总计  : {total_covered} ({total_covered / total_logs:.1%})")
        print("=" * 60 + "\n")


        return sorted(anomalies, key=lambda x: -x['score'])

    def get_sample_logs(self, log_ids, max_samples=3):
        """优化版获取样本日志，选择差异度最大的日志"""

        def tokenize(log):
            # 基础分词逻辑，可根据需要调整
            return set(re.findall(r'\S+', log.lower()))

        def jaccard_distance(a, b):
            set_a = tokenize(a)
            set_b = tokenize(b)
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            return 1 - intersection / union if union != 0 else 0

        try:
            # 获取日志内容并去重
            # 改进点：随机采样代替顺序采样
            sample_size = min(1000, len(log_ids))  # 防止列表长度不足
            sampled_ids = random.sample(log_ids, sample_size) if log_ids else []

            # 获取日志内容并去重
            logs = self.df_log.loc[
                self.df_log['LineId'].isin(sampled_ids),
                'Content'
            ].dropna().unique()

            if not logs.size:
                return []

            logs = [str(log) for log in logs]
            selected = []

            # 初始选择最长日志作为种子
            if logs:
                seed = max(logs, key=lambda x: len(x))
                selected.append(seed)
                logs.remove(seed)

            # 最大最小距离法选择差异样本
            while len(selected) < max_samples and logs:
                max_min_dist = -1
                best_candidate = None

                for candidate in logs:
                    # 计算与已选日志的最小距离
                    min_dist = min(jaccard_distance(candidate, s) for s in selected)
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_candidate = candidate

                if best_candidate:
                    selected.append(best_candidate)
                    logs.remove(best_candidate)
                else:
                    break

            return selected[:max_samples]

        except Exception as e:
            print(f"获取样本日志时出错: {str(e)}")
            return []

    def save_anomalous_templates(self, anomalies):
        """增强版结果保存，集成数据集名称"""
        if not anomalies:
            print("未检测到异常模板")
            return

        # 构建详细报告
        report = []
        for anomaly in anomalies:
            record = {
                'Dataset': self.logName,  # 新增数据集名称字段
                'Template': anomaly['template'],
                'AnomalyScore': anomaly['score'],
                'WildcardRatio': f"{anomaly['metrics']['wildcard_ratio'] * 100:.1f}%",
                'ConsecutiveWildcards': f"{anomaly['metrics']['consecutive_wildcards'] * 100:.1f}%",
                'WildcardDensity': f"{anomaly['metrics']['wildcard_density'] * 100:.1f}%",
                'SymbolDensity': anomaly['metrics']['symbol_density'],
                'NumericDensity': anomaly['metrics']['numeric_density'],
                **{f'Sample{i + 1}': sample for i, sample in enumerate(anomaly['samples'])}
            }
            report.append(record)

        # 创建统一保存路径
        save_dir = os.path.join(self.savePath, "anomaly_reports")
        os.makedirs(save_dir, exist_ok=True)

        # 使用数据集名称+时间戳命名（参考cluster保存方式）
        base_name = f"{self.logName}_anomaly_{datetime.now().strftime('%Y%m%d_%H%M')}"

        # CSV版本
        csv_path = os.path.join(save_dir, f"{base_name}.csv")
        pd.DataFrame(report).to_csv(csv_path, index=False)

        # JSON版本（增强可读性）
        json_path = os.path.join(save_dir, f"{base_name}.json")
        enhanced_data = {
            "dataset": self.logName,
            "generated_time": datetime.now().isoformat(),
            "anomaly_count": len(anomalies),
            "anomalies": anomalies
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, ensure_ascii=False, indent=2)

        print(f"异常报告已保存：\nCSV -> {csv_path}\nJSON -> {json_path}")

class ClusterRefiner:
    """基于大模型的日志模板聚类优化器"""

    SYSTEM_PROMPT = (
        """You are a log parsing expert. 
    Analyze log messages to generate correct templates by replacing variable values with <*> while preserving constant patterns. 
    Study the following examples to reparse similar templates：
    
        Log List 1:
    [
        "Processing request from user_123 with parameters 100 200 300",
        "Processing request from user_456 with parameters 700 800 900"
    ]
    
    Log List 2:
    [
        "for url=13562/mapOutput?job=job_1445087491445_0004&reduce=0&map=attempt_1445087491445_0004_m_000012_1,attempt_1445087491445_0004_m_000004_1000,attempt_1445087491445_0004_m_000009_1000,attempt_1445087491445_0004_m_000002_1000 sent hash and received reply",
        "for url=13562/mapOutput?job=job_1445087491445_0005&reduce=0&map=attempt_1445087491445_0005_m_000012_1,attempt_1445087491445_0005_m_000004_1000,attempt_1445087491445_0005_m_000009_1000,attempt_1445087491445_0005_m_000002_1000 sent hash and received reply"
    ]
    
    Log List 3:
    [
        "Task: attempt_1445144423722_0020_m_000002_0 - exited : java.net.NoRouteToHostException: No Route to Host from  MININT-FNANLI5/127.0.0.1 to msra-sa-41:9000 failed on socket timeout exception: java.net.NoRouteToHostException: No route to host: no further information; For more details see:  http://wiki.apache.org/hadoop/NoRouteToHost",
        "Task: attempt_1445144423722_0021_m_000002_0 - exited : java.net.NoRouteToHostException: No Route to Host from  MININT-FNANLI5/127.0.0.1 to msra-sa-42:9000 failed on socket timeout exception: java.net.NoRouteToHostException: No route to host: no further information; For more details see:  http://wiki.apache.org/hadoop/NoRouteToHost"
    ]

    Correct output format:
    Log template1: Processing request from <*> with parameters <*>
    Log template2: for url=<*> sent hash and received reply
    Log template3: Task: <*> - exited : java.net.NoRouteToHostException: No Route to Host from <*> to <*> failed on socket timeout exception: java.net.NoRouteToHostException: No route to host: no further information; For more details see: <*>
    """
    )

    API_CONFIG = {
        "endpoint": "https://api.deepseek.com",
        "api_key": "sk-5503aefefd0b404f8d7e9dd6835513da",
        "model": "deepseek-chat"
    }

    def __init__(self):
        self.client = OpenAI(
            api_key=self.API_CONFIG["api_key"],
            base_url=self.API_CONFIG["endpoint"]
        )

    def refine_clusters(self, raw_clusters, clustered_groups):
        """
        优化并合并raw_clusters：针对每组clustered_groups调用API，
        将所有API解析后的模板更新到对应logclusters，
        并合并相同解析结果的所有logclusters（包括未分组中的匹配项），
        未参与解析的clusters原样保留
        """
        parsed_map = defaultdict(list)  # parsed_template_str -> List[Logcluster]
        processed_ids = set()

        # 1. 调用API解析每个分组的模板
        for group in clustered_groups:
            prompt_parts = []
            for idx, tpl in enumerate(group['templates'], start=1):
                # 获取前两个样本（不足则取全部）
                samples = [re.sub(r'\s+', ' ', s).strip() for s in tpl['samples'][:2]] if tpl['samples'] else ['(No sample)']
                # 添加Log List和模板信息
                prompt_parts.append(f"Log List {idx}:")
                prompt_parts.append(json.dumps(samples, ensure_ascii=False))
            user_prompt = "\n".join(prompt_parts)

            print("=================API查询构造=====================")
            print(user_prompt)

            response = self.client.chat.completions.create(
                model=self.API_CONFIG['model'],
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            num_templates = len(group['templates'])  # 获取当前group的模板数量
            refined = self._parse_response(response.choices[0].message.content, num_templates)

            # 更新每个group内部的clusters
            for idx, tpl in enumerate(group['templates'], start=1):
                key = f"Log template{idx}"
                # 在更新模板前添加校验
                if idx > len(refined):
                    print(f"警告：分组{group['cluster_id']}的模板{idx}缺少API响应，使用最后模板补齐")
                    parsed_str = list(refined.values())[-1] if refined else tpl['template']
                else:
                    parsed_str = refined.get(key, tpl['template'])
                # 查找对应raw_clusters并更新
                for cluster in raw_clusters:
                    if cluster.uuid == tpl['template_id']:
                        cluster.logTemplate = parsed_str.split()
                        cluster.parsed_by_llm = True
                        parsed_map[parsed_str].append(cluster)
                        processed_ids.add(cluster.uuid)
                        break

        # 2. 合并API解析结果与raw_clusters中匹配但未分组的templates
        for parsed_str, clusters in list(parsed_map.items()):
            # 获取解析后的模板对象
            parsed_template = ' '.join(clusters[0].logTemplate)

            # 遍历所有未处理原始集群
            for cluster in raw_clusters:
                if cluster.uuid in processed_ids:
                    continue

                # 获取当前原始模板
                raw_template = ' '.join(cluster.logTemplate)

                # 尝试合并模板
                merged_template = self.merge_log_templates(parsed_template, raw_template)

                if merged_template:
                    print(f"合并成功: {parsed_template} + {raw_template} => {merged_template}")
                    # 同时更新两个集群的模板
                    for c in clusters:
                        c.logTemplate = merged_template.split()
                    cluster.logTemplate = merged_template.split()
                    # 标记为已处理
                    cluster.parsed_by_llm = True
                    clusters.append(cluster)
                    processed_ids.add(cluster.uuid)

        # 3. 对parsed_map中每个解析结果分组进行合并
        merged = []
        for parsed_str, groups in parsed_map.items():
            main = groups[0]
            # 确保LLM解析标记
            main.parsed_by_llm = True
            for c in groups[1:]:
                main.logIDL.extend(c.logIDL)
            main.logTemplate = parsed_str.split()
            merged.append(main)

        # 4. 保留未参与任何解析的原始clusters
        rest = [c for c in raw_clusters if c.uuid not in processed_ids]
        return rest + merged

    def tokenize(self, s: str) -> List[str]:
        # 分割规则：空格、括号、等号
        SPLIT_PATTERN = r'(\s+|\(|\)|\[|\]|=)'

        """按照空格、括号、等号切分，并保留分隔符"""
        tokens = re.split(SPLIT_PATTERN, s)
        return [token for token in tokens if token != '']

    def is_wildcard(self, token: str) -> bool:
        return token == '<*>'

    def merge_two_sequences(self, seq1: List[str], seq2: List[str]) -> Tuple[bool, List[str]]:
        """
        合并两个模板序列，如果能合并则返回True和合并后的序列，否则返回False
        """
        # 长度限制(<=3)
        MAX_WILDCARD_MATCH_LENGTH = 2

        matcher = SequenceMatcher(None, seq1, seq2)
        result = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            part1 = seq1[i1:i2]
            part2 = seq2[j1:j2]

            if tag == 'equal':
                result.extend(part1)
            else:
                # 合并逻辑
                if part1 == part2:
                    result.extend(part1)
                # 先检查空列表和通配符的情况
                elif not part1 and part2 and all(self.is_wildcard(t) for t in part2):
                    result.extend(part2)
                elif not part2 and part1 and all(self.is_wildcard(t) for t in part1):
                    result.extend(part1)
                # 再检查通配符和长度限制的情况
                elif part1 and all(self.is_wildcard(t) for t in part1) and len(part2) <= MAX_WILDCARD_MATCH_LENGTH:
                    result.extend(part1)
                elif part2 and all(self.is_wildcard(t) for t in part2) and len(part1) <= MAX_WILDCARD_MATCH_LENGTH:
                    result.extend(part2)
                else:
                    return False, []
        return True, result

    def merge_log_templates(self, template1: str, template2: str) -> str | None:
        """
        合并两个日志模板字符串，返回合并后的结果（失败时返回None）

        Args:
            template1: 第一个日志模板字符串，如 "Received disconnect from <*> : <*> : disconnect [ preauth ]"
            template2: 第二个日志模板字符串，如 "Received disconnect from <*> : <*> : [ preauth ]"

        Returns:
            str: 合并后的模板字符串（如能合并）
            None: 合并失败时返回
        """
        # 分词处理
        seq1 = self.tokenize(template1)
        seq2 = self.tokenize(template2)

        # 尝试合并
        success, merged_tokens = self.merge_two_sequences(seq1, seq2)

        # 重组为字符串
        return ''.join(merged_tokens) if success else None

    def merge_templates_dict(self, templates: Dict[str, str]) -> Dict[str, str]:
        """
        合并模板，成功合并的模板使用新结果，未合并的保留原值
        """
        original = templates.copy()
        tokenized = {k: self.tokenize(v) for k, v in templates.items()}
        if not tokenized:
            return original

        # 计算通配符数量辅助函数
        def count_wildcards(tokens):
            return sum(1 for t in tokens if self.is_wildcard(t))

        # 基准模板排序策略
        sorted_keys = sorted(
            tokenized.keys(),
            key=lambda k: (-count_wildcards(tokenized[k]), len(tokenized[k]))
        )

        # 初始化基准模板和已合并集合
        base_key = sorted_keys[0]
        base = tokenized[base_key].copy()
        merged_keys = {base_key}

        # 剩余模板排序策略
        remaining_keys = sorted(
            sorted_keys[1:],
            key=lambda k: (-len(tokenized[k]), -count_wildcards(tokenized[k]))
        )

        # 逐步合并模板
        for key in remaining_keys:
            ok, new_base = self.merge_two_sequences(base, tokenized[key])
            if ok:
                base = new_base
                merged_keys.add(key)

        # 构造最终结果
        merged_str = ''.join(base)
        return {
            k: merged_str if k in merged_keys else original[k]
            for k in templates.keys()
        }

    def correct_single_template(self, template, user_strings=None):
        """Apply all rules to process a template.

        DS (Double Space)
        BL (Boolean) # we don't use this
        US (User String) # we don't use this
        DG (Digit)
        PS (Path-like String) # we don't use this
        WV (Word concatenated with Variable)
        DV (Dot-separated Variables)
        CV (Consecutive Variables)

        """

        # boolean = {}
        # default_strings = {}
        path_delimiters = {  # reduced set of delimiters for tokenizing for checking the path-like strings
            r'\s', r'\,', r'\!', r'\;', r'\:',
            r'\=', r'\|', r'\"', r'\'',
            r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
        }
        token_delimiters = path_delimiters.union({  # all delimiters for tokenizing the remaining rules
            r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&',
        })

        # if user_strings:
        # default_strings = default_strings.union(user_strings)

        # apply DS
        template = template.strip()
        template = re.sub(r'\s+', ' ', template)

        # apply PS
        # p_tokens = re.split('(' + '|'.join(path_delimiters) + ')', template)
        # new_p_tokens = []
        # for p_token in p_tokens:
        # if re.match(r'^(\/[^\/]+)+$', p_token):
        # p_token = '<*>'
        # new_p_tokens.append(p_token)
        # template = ''.join(new_p_tokens)

        # tokenize for the remaining rules
        tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)  # tokenizing while keeping delimiters
        new_tokens = []
        for token in tokens:
            # apply BL, US
            # for to_replace in boolean.union(default_strings):
            # if token.lower() == to_replace.lower():
            # token = '<*>'

            # apply DG
            if re.match(r'^(?=.*\d)(?!.*[A-Za-z])[\d\W_]+$', token):
                token = re.sub(r'\d+', '<*>', token)

            # apply WV
            if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token):
                if token != '<*>/<*>':  # need to check this because `/` is not a deliminator
                    token = '<*>'

            # collect the result
            new_tokens.append(token)

        # make the template using new_tokens
        template = ''.join(new_tokens)

        # Substitute consecutive variables only if separated with any delimiter including "." (DV)
        while True:
            prev = template
            template = re.sub(r'<\*>\.<\*>', '<*>', template)
            if prev == template:
                break

        # Substitute consecutive variables only if not separated with any delimiter including space (CV)
        # NOTE: this should be done at the end
        # print("CV: ", template)
        while True:
            prev = template
            template = re.sub(r'<\*><\*>', '<*>', template)
            if prev == template:
                break
        # print("CV: ", template)

        while " #<*># " in template:
            template = template.replace(" #<*># ", " <*> ")

        while " #<*> " in template:
            template = template.replace(" #<*> ", " <*> ")

        while "<*>:<*>" in template:
            template = template.replace("<*>:<*>", "<*>")

        while "<*>#<*>" in template:
            template = template.replace("<*>#<*>", "<*>")

        while "<*>/<*>" in template:
            template = template.replace("<*>/<*>", "<*>")

        while "<*>@<*>" in template:
            template = template.replace("<*>@<*>", "<*>")

        while "<*>.<*>" in template:
            template = template.replace("<*>.<*>", "<*>")

        while ' "<*>" ' in template:
            template = template.replace(' "<*>" ', ' <*> ')

        while " '<*>' " in template:
            template = template.replace(" '<*>' ", " <*> ")

        while "<*><*>" in template:
            template = template.replace("<*><*>", "<*>")
        return template

    def _parse_response(self, text, expected_templates=0):
        templates = {}
        counter = 1
        last_template = None

        print("=================API响应结果=====================")
        print(text)

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            # 匹配模板格式
            m = re.match(r"(?i)(Log template\d+):\s*(.*)", line)
            if m:
                key = m.group(1)
                value = self.correct_single_template(m.group(2).strip())  # 新增清洗步骤
                templates[key] = value
                last_template = value
                counter += 1
            elif line.startswith("<"):
                # 当检测到通配符模板但未匹配正则时
                cleaned = self.correct_single_template(line)  # 新增清洗步骤
                templates[f"Log template{counter}"] = cleaned
                last_template = cleaned
                counter += 1

        # 处理模板数量不足的情况（核心逻辑）
        if expected_templates > 0:
            # 当API返回数量不足时
            if len(templates) < expected_templates and last_template is not None:
                for i in range(len(templates) + 1, expected_templates + 1):
                    key = f"Log template{i}"
                    if key not in templates:
                        # 对补全的模板也进行清洗
                        templates[key] = self.correct_single_template(last_template)
            # 当API返回数量超过时取前N个
            elif len(templates) > expected_templates:
                templates = {k: self.correct_single_template(v)
                             for k, v in list(templates.items())[:expected_templates]}

        templates = self.merge_templates_dict(templates)

        print("=================清洗API结果=====================")
        for key, value in templates.items():
            print(f"{key}: {value}")

        return templates


class AnomalyRefiner:
    """基于大模型的异常模板解析与合并优化器"""

    SYSTEM_PROMPT = (
        """You are a log parsing expert. Analyze log messages to generate correct templates by replacing variable values with <*> while preserving constant patterns. Follow these examples:

    Log List 1:
    [
        "Processing request from user_123 with parameters 100 200 300",
        "Processing request from user_456 with parameters 700 800 900"
    ]
    
    Log List 2:
    [
        "for url=13562/mapOutput?job=job_1445087491445_0004&reduce=0&map=attempt_1445087491445_0004_m_000012_1,attempt_1445087491445_0004_m_000004_1000,attempt_1445087491445_0004_m_000009_1000,attempt_1445087491445_0004_m_000002_1000 sent hash and received reply",
        "for url=13562/mapOutput?job=job_1445087491445_0005&reduce=0&map=attempt_1445087491445_0005_m_000012_1,attempt_1445087491445_0005_m_000004_1000,attempt_1445087491445_0005_m_000009_1000,attempt_1445087491445_0005_m_000002_1000 sent hash and received reply"
    ]
    
    Log List 3:
    [
        "Task: attempt_1445144423722_0020_m_000002_0 - exited : java.net.NoRouteToHostException: No Route to Host from  MININT-FNANLI5/127.0.0.1 to msra-sa-41:9000 failed on socket timeout exception: java.net.NoRouteToHostException: No route to host: no further information; For more details see:  http://wiki.apache.org/hadoop/NoRouteToHost",
        "Task: attempt_1445144423722_0021_m_000002_0 - exited : java.net.NoRouteToHostException: No Route to Host from  MININT-FNANLI5/127.0.0.1 to msra-sa-42:9000 failed on socket timeout exception: java.net.NoRouteToHostException: No route to host: no further information; For more details see:  http://wiki.apache.org/hadoop/NoRouteToHost"
    ]
    
    Correct output format:
    Log Template 1: Processing request from <*> with parameters <*>
    Log Template 2: for url=<*> sent hash and received reply
    Log Template 3: Task: <*> - exited : java.net.NoRouteToHostException: No Route to Host from <*> to <*> failed on socket timeout exception: java.net.NoRouteToHostException: No route to host: no further information; For more details see: <*>


Now process this message:
"""
    )

    # 复用 ClusterRefiner 中的 API 配置
    API_CONFIG = {
        "endpoint": "https://api.deepseek.com",
        "api_key": "sk-e00c3df34c69433d9a289a77a2880daa",
        "model": "deepseek-chat"
    }

    def __init__(self):
        self.client = OpenAI(
            api_key=self.API_CONFIG["api_key"],
            base_url=self.API_CONFIG["endpoint"]
        )

    def refine_anomalies(self, log_clusters, anomalous_templates):
        """
        接受原始 log_clusters 列表及 detect_anomalous_templates 返回的 anomalous_templates 列表，
        1. 针对每个异常模板调用 LLM 解析，更新 logTemplate 并标记 parsed_by_llm=True；
        2. 若解析结果与 log_clusters 中某现有 cluster 模板相同，则合并它们的 logIDL；
        3. 返回合并及更新后的完整 log_clusters 列表。
        """

        parsed_map = defaultdict(list)      # refined_str -> [Logcluster...]
        processed_ids = set()

        # 一、对每个异常模板调用 API 解析
        for tpl in anomalous_templates:
            # Extract samples as a Log List
            samples = tpl['samples'][:2]  # Take up to 3 samples for better learning
            log_list_example = json.dumps(samples, ensure_ascii=False)

            prompt = f"Log List:\n{log_list_example}"

            print("=================API查询构造=====================")
            print(prompt)

            resp = self.client.chat.completions.create(
                model=self.API_CONFIG["model"],
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            refined = self._parse_single_response(resp.choices[0].message.content)

            print("=================清洗API结果=====================")
            print(refined)

            # 在log_clusters中查找匹配的Logcluster对象
            target_cluster = next(
                (cluster for cluster in log_clusters
                 if ' '.join(cluster.logTemplate) == tpl['template']),
                None
            )

            if target_cluster:
                # 更新找到的Logcluster对象
                target_cluster.logTemplate = refined.split()
                target_cluster.parsed_by_llm = True
                parsed_map[refined].append(target_cluster)
                processed_ids.add(target_cluster.uuid)

        # 二、合并解析结果与现有 clusters 中模板一致的项
        for refined_str, group in list(parsed_map.items()):
            # 获取解析后的模板对象
            parsed_template = ' '.join(group[0].logTemplate)

            # 遍历所有未处理原始集群
            for cluster in log_clusters:
                if cluster.uuid in processed_ids:
                    continue

                # 获取当前原始模板
                raw_template = ' '.join(cluster.logTemplate)

                # 尝试合并模板
                merged_template = self.merge_log_templates(parsed_template, raw_template)

                if merged_template:
                    print(f"合并成功: {parsed_template} + {raw_template} => {merged_template}")
                    # 同时更新两个集群的模板
                    for c in group:
                        c.logTemplate = merged_template.split()
                    cluster.logTemplate = merged_template.split()
                    # 标记为已处理
                    cluster.parsed_by_llm = True
                    group.append(cluster)
                    processed_ids.add(cluster.uuid)

        # 三、根据解析结果合并各组 clusters
        merged = []
        for refined_str, group in parsed_map.items():
            main = group[0]
            main.parsed_by_llm = True
            for other in group[1:]:
                main.logIDL.extend(other.logIDL)
            main.logTemplate = refined_str.split()
            merged.append(main)

        # 四、保留未参与解析/合并的剩余 clusters
        rest = [c for c in log_clusters if c.uuid not in processed_ids]
        return rest + merged

    def _parse_single_response(self, text):
        print("=================API响应结果=====================")
        print(text)

        # 尝试跨行匹配 Log Template 或 Template 行
        match = re.search(r"(?i)(?:Log[ _]?Template|Template)\s*:\s*([^\n\r]+)", text)
        if match:
            parsed = match.group(1).strip()
            # 白名单关键词直接放行
            if any(kw in parsed.lower() for kw in {"block report", "screen off"}):
                return parsed
            return self.correct_single_template(parsed)

        # 回退到逐行查找
        for line in text.splitlines():
            line = line.strip()
            if re.match(r"(?i)(?:Log[ _]?Template|Template):", line):
                parsed = re.sub(r'.*?:\s*', '', line).strip()
                return self.correct_single_template(parsed)

        # 最终回退
        for line in text.splitlines():
            if line.strip() and "<*>" in line:
                return self.correct_single_template(line.strip())
        return text.splitlines()[0].strip() if text.splitlines() else ""

    def correct_single_template(self, template, user_strings=None):
        """Apply all rules to process a template.

        DS (Double Space)
        BL (Boolean) # we don't use this
        US (User String) # we don't use this
        DG (Digit)
        PS (Path-like String) # we don't use this
        WV (Word concatenated with Variable)
        DV (Dot-separated Variables)
        CV (Consecutive Variables)

        """

        # boolean = {}
        # default_strings = {}
        path_delimiters = {  # reduced set of delimiters for tokenizing for checking the path-like strings
            r'\s', r'\,', r'\!', r'\;', r'\:',
            r'\=', r'\|', r'\"', r'\'',
            r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
        }
        token_delimiters = path_delimiters.union({  # all delimiters for tokenizing the remaining rules
            r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&',
        })

        # if user_strings:
        # default_strings = default_strings.union(user_strings)

        # apply DS
        template = template.strip()
        template = re.sub(r'\s+', ' ', template)

        # apply PS
        # p_tokens = re.split('(' + '|'.join(path_delimiters) + ')', template)
        # new_p_tokens = []
        # for p_token in p_tokens:
        # if re.match(r'^(\/[^\/]+)+$', p_token):
        # p_token = '<*>'
        # new_p_tokens.append(p_token)
        # template = ''.join(new_p_tokens)

        # tokenize for the remaining rules
        tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)  # tokenizing while keeping delimiters
        new_tokens = []
        for token in tokens:
            # apply BL, US
            # for to_replace in boolean.union(default_strings):
            # if token.lower() == to_replace.lower():
            # token = '<*>'

            # apply DG
            if re.match(r'^(?=.*\d)(?!.*[A-Za-z])[\d\W_]+$', token):
                token = re.sub(r'\d+', '<*>', token)

            # apply WV
            if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token):
                if token != '<*>/<*>':  # need to check this because `/` is not a deliminator
                    token = '<*>'

            # collect the result
            new_tokens.append(token)

        # make the template using new_tokens
        template = ''.join(new_tokens)

        # Substitute consecutive variables only if separated with any delimiter including "." (DV)
        while True:
            prev = template
            template = re.sub(r'<\*>\.<\*>', '<*>', template)
            if prev == template:
                break

        # Substitute consecutive variables only if not separated with any delimiter including space (CV)
        # NOTE: this should be done at the end
        # print("CV: ", template)
        while True:
            prev = template
            template = re.sub(r'<\*><\*>', '<*>', template)
            if prev == template:
                break
        # print("CV: ", template)

        while " #<*># " in template:
            template = template.replace(" #<*># ", " <*> ")

        while " #<*> " in template:
            template = template.replace(" #<*> ", " <*> ")

        while "<*>:<*>" in template:
            template = template.replace("<*>:<*>", "<*>")

        while "<*>#<*>" in template:
            template = template.replace("<*>#<*>", "<*>")

        while "<*>/<*>" in template:
            template = template.replace("<*>/<*>", "<*>")

        while "<*>@<*>" in template:
            template = template.replace("<*>@<*>", "<*>")

        while "<*>.<*>" in template:
            template = template.replace("<*>.<*>", "<*>")

        while ' "<*>" ' in template:
            template = template.replace(' "<*>" ', ' <*> ')

        while " '<*>' " in template:
            template = template.replace(" '<*>' ", " <*> ")

        while "<*><*>" in template:
            template = template.replace("<*><*>", "<*>")
        return template


    def tokenize(self, s: str) -> List[str]:
        # 分割规则：空格、括号、等号
        SPLIT_PATTERN = r'(\s+|\(|\)|\[|\]|=)'

        """按照空格、括号、等号切分，并保留分隔符"""
        tokens = re.split(SPLIT_PATTERN, s)
        return [token for token in tokens if token != '']

    def is_wildcard(self, token: str) -> bool:
        return token == '<*>'

    def merge_two_sequences(self, seq1: List[str], seq2: List[str]) -> Tuple[bool, List[str]]:
        """
        合并两个模板序列，如果能合并则返回True和合并后的序列，否则返回False
        """
        # 长度限制(<=3)
        MAX_WILDCARD_MATCH_LENGTH = 2

        matcher = SequenceMatcher(None, seq1, seq2)
        result = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            part1 = seq1[i1:i2]
            part2 = seq2[j1:j2]

            if tag == 'equal':
                result.extend(part1)
            else:
                # 合并逻辑
                if part1 == part2:
                    result.extend(part1)
                # 先检查空列表和通配符的情况
                elif not part1 and part2 and all(self.is_wildcard(t) for t in part2):
                    result.extend(part2)
                elif not part2 and part1 and all(self.is_wildcard(t) for t in part1):
                    result.extend(part1)
                # 再检查通配符和长度限制的情况
                elif part1 and all(self.is_wildcard(t) for t in part1) and len(part2) <= MAX_WILDCARD_MATCH_LENGTH:
                    result.extend(part1)
                elif part2 and all(self.is_wildcard(t) for t in part2) and len(part1) <= MAX_WILDCARD_MATCH_LENGTH:
                    result.extend(part2)
                else:
                    return False, []
        return True, result

    def merge_log_templates(self, template1: str, template2: str) -> str | None:
        """
        合并两个日志模板字符串，返回合并后的结果（失败时返回None）

        Args:
            template1: 第一个日志模板字符串，如 "Received disconnect from <*> : <*> : disconnect [ preauth ]"
            template2: 第二个日志模板字符串，如 "Received disconnect from <*> : <*> : [ preauth ]"

        Returns:
            str: 合并后的模板字符串（如能合并）
            None: 合并失败时返回
        """
        # 分词处理
        seq1 = self.tokenize(template1)
        seq2 = self.tokenize(template2)

        # 尝试合并
        success, merged_tokens = self.merge_two_sequences(seq1, seq2)

        # 重组为字符串
        return ''.join(merged_tokens) if success else None

    def merge_templates_dict(self, templates: Dict[str, str]) -> Dict[str, str]:
        """
        合并模板，成功合并的模板使用新结果，未合并的保留原值
        """
        original = templates.copy()
        tokenized = {k: self.tokenize(v) for k, v in templates.items()}
        if not tokenized:
            return original

        # 计算通配符数量辅助函数
        def count_wildcards(tokens):
            return sum(1 for t in tokens if self.is_wildcard(t))

        # 基准模板排序策略
        sorted_keys = sorted(
            tokenized.keys(),
            key=lambda k: (-count_wildcards(tokenized[k]), len(tokenized[k]))
        )

        # 初始化基准模板和已合并集合
        base_key = sorted_keys[0]
        base = tokenized[base_key].copy()
        merged_keys = {base_key}

        # 剩余模板排序策略
        remaining_keys = sorted(
            sorted_keys[1:],
            key=lambda k: (-len(tokenized[k]), -count_wildcards(tokenized[k]))
        )

        # 逐步合并模板
        for key in remaining_keys:
            ok, new_base = self.merge_two_sequences(base, tokenized[key])
            if ok:
                base = new_base
                merged_keys.add(key)

        # 构造最终结果
        merged_str = ''.join(base)
        return {
            k: merged_str if k in merged_keys else original[k]
            for k in templates.keys()
        }