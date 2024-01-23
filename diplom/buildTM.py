from sys import path
if r'C:\BigARTM\Python' not in path:
    path.append(r'C:\BigARTM\Python')

import os
import glob
import artm
import time
import matplotlib.pyplot as plt

DATA_PATH = r'habr\habr_posts\no_code\texts135k.txt'
QUERY = r'habr\habr_posts\no_code\query.txt'
BATCHES = 'habr_batches'
QUERY_BATCHES = 'habr_query_batches'

def save_phi_theta(model, save_phi=False, save_theta=True):
    if save_phi:
        model.get_phi().transpose().to_hdf('phi_store.h5', 'table', mode='w')
    if save_theta:
        model.get_theta().transpose().to_hdf('theta_store.h5', 'table', mode='w')

def get_query_data(model, predict_class=None, query_file=QUERY, save_theta_q=True,
                   save_classes=True, filename='thetaQ_store'):
    query_bv = artm.BatchVectorizer(data_path=query_file,
                                    data_format='vowpal_wabbit',
                                    batch_size=10,
                                    target_folder=QUERY_BATCHES,
                                    gather_dictionary=True)

    thetaQ = model.transform(batch_vectorizer=query_bv)
    if predict_class:
        classes = model.transform(batch_vectorizer=query_bv, predict_class_id=predict_class)
    if save_theta_q:
        thetaQ.transpose().to_hdf(filename + '.h5', 'table')
    if predict_class is not None and save_classes:
        classes.sort_values(classes.columns[0]).to_csv(filename + '_classes.csv', sep='\t', encoding='utf-8')

def get_tokens(model, flname='res.txt'):
    tokens = model.score_tracker['top_tok'].last_tokens
    with open(flname, 'w', encoding='utf-8') as fl:
        for topic_name in model.topic_names:
            print(topic_name, file=fl)
            for token in tokens[topic_name]:
                print(token, end=' ', file=fl)
            print(file=fl)

def draw_data(model, ConPur=False):
    perp_values = model.score_tracker['PerplAll'].value[1:]
    perp_values_w = model.score_tracker['PerplWords'].value[1:]
    sparsity_phi = model.score_tracker['SparPhiAll'].value
    sparsity_phi_main = model.score_tracker['SparPhiMain'].value
    sparsity_phi_bckgr = model.score_tracker['SparPhiBckgr'].value
    sparsity_theta = model.score_tracker['SparThetAll'].value
    sparsity_theta_main = model.score_tracker['SparThetMain'].value
    sparsity_theta_bckgr = model.score_tracker['SparThetBckgr'].value

    plt.ion()
    # Perp
    fig, axs = plt.subplots(3, 2)
    fig.suptitle('Перплексия')
    axs[0, 0].set_title('All', fontsize=10)
    axs[0, 1].set_title('Words', fontsize=10)
    for i in range(3):
        axs[i, 0].plot(range(len(perp_values[i * 5:])), perp_values[i * 5:])
        axs[i, 0].grid(True)
        axs[i, 1].plot(range(len(perp_values_w[i * 5:])), perp_values_w[i * 5:])
        axs[i, 1].grid(True)
    plt.show()
    # SparPhi
    fig, axs = plt.subplots(3)
    fig.suptitle('SparPhi')
    axs[0].plot(range(len(sparsity_phi)), sparsity_phi)
    axs[1].plot(range(len(sparsity_phi_main)), sparsity_phi_main)
    axs[2].plot(range(len(sparsity_phi_bckgr)), sparsity_phi_bckgr)
    for i in range(3):
        axs[i].grid(True)
    plt.show()
    # SparTheta
    fig, axs = plt.subplots(3)
    fig.suptitle('SparTheta')
    axs[0].plot(range(len(sparsity_theta)), sparsity_theta)
    axs[1].plot(range(len(sparsity_theta_main)), sparsity_theta_main)
    axs[2].plot(range(len(sparsity_theta_bckgr)), sparsity_theta_bckgr)
    for i in range(3):
        axs[i].grid(True)
    plt.show()
    if ConPur:
        contrast = model.score_tracker['TopicKernelScore'].average_contrast
        purity = model.score_tracker['TopicKernelScore'].average_purity
        plt.figure(num=4)
        plt.plot(range(len(contrast)), contrast)
        plt.title('Contrast')
        plt.grid(True)
        plt.figure(num=5)
        plt.plot(range(len(purity)), purity)
        plt.title('Purity')
        plt.grid(True)
        plt.show()

try:
    if len(glob.glob(os.path.join(BATCHES, '*.batch'))) == 0:
        start_bv = time.time()
        bv = artm.BatchVectorizer(data_path=DATA_PATH,
                                  data_format='vowpal_wabbit',
                                  batch_size=1000,
                                  target_folder=BATCHES,
                                  gather_dictionary=True)
        end_bv = time.time() - start_bv
        print('Batches in {0:.0f}m{1:.0f}s'.format((end_bv // 60) % 60, end_bv % 60, ))
        dictionary = bv.dictionary
    else:
        bv = artm.BatchVectorizer(data_path=BATCHES,
                                  data_format='batches',
                                  gather_dictionary=True)
        dictionary = artm.Dictionary()
        dictionary.load(dictionary_path=os.path.join(BATCHES, 'dictionary.dict'))

    if not os.path.isfile(os.path.join(BATCHES, 'dictionary.dict')):
        dictionary.gather(data_path=bv.data_path)
        dictionary.save(dictionary_path=os.path.join(BATCHES, 'dictionary.dict'))

    topics = ['topic_' + str(i) for i in range(250)]
    background_topics = topics[:50]
    main_topics = topics[50:]

    doc_passes = 5

    model = artm.ARTM(topic_names=topics,
                      cache_theta=True,
                      dictionary=dictionary,
                      show_progress_bars=True,
                      num_document_passes=doc_passes,
                      class_ids={'words': 1.0, 'hubs': 10.0},
                      theta_columns_naming='title',
                      theta_name='Habr_theta')
    ## Regularaizers
    # SSPhi
    model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SSPhiMain', topic_names=main_topics, tau=0.01))
    model.regularizers.add(
        artm.SmoothSparsePhiRegularizer(name='SSPhiBckgr', topic_names=background_topics, tau=0.015))
    # SSTheta
    model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SSThetaMain', topic_names=main_topics, tau=0.01))
    model.regularizers.add(
        artm.SmoothSparseThetaRegularizer(name='SSThetaBckgr', topic_names=background_topics, tau=0.015))
    # DecorrelatorPhi
    model.regularizers.add(
        artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhiMain', topic_names=main_topics, tau=1e+3))
    model.regularizers.add(
        artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhiBckgr', topic_names=background_topics, tau=1e+2))

    ## Scores
    # Perplexity
    model.scores.add(artm.PerplexityScore(name='PerplAll', dictionary=bv.dictionary, ))
    model.scores.add(artm.PerplexityScore(name='PerplWords', class_ids='words', dictionary=bv.dictionary, ))
    # SparPhi
    model.scores.add(artm.SparsityPhiScore(name='SparPhiAll', class_id='words'))
    model.scores.add(artm.SparsityPhiScore(name='SparPhiMain', topic_names=main_topics, class_id='words'))
    model.scores.add(artm.SparsityPhiScore(name='SparPhiBckgr', topic_names=background_topics, class_id='words'))
    # SparThet
    model.scores.add(artm.SparsityThetaScore(name='SparThetAll'))
    model.scores.add(artm.SparsityThetaScore(name='SparThetMain', topic_names=main_topics))
    model.scores.add(artm.SparsityThetaScore(name='SparThetBckgr', topic_names=background_topics))
    # Other
    model.scores.add(artm.TopTokensScore(name='top_tok', num_tokens=10, class_id='words'))
    model.scores.add(artm.TopicKernelScore(name='TopicKernelScore',
                                           class_id='words',
                                           topic_names=main_topics,
                                           probability_mass_threshold=0.5))
    ## Fitting
    model.fit_offline(batch_vectorizer=bv, num_collection_passes=15)
    model.dump_artm_model('250_topics_15itr')
    get_tokens(model, 'res1.txt')

    model.regularizers['SSThetaMain'].tau = 0
    model.regularizers['SSPhiMain'].tau = 0
    model.regularizers['DecorrelatorPhiMain'].tau = 1e+4

    model.fit_offline(batch_vectorizer=bv, num_collection_passes=20)
    model.dump_artm_model('250_topics_35itr')
    get_tokens(model, 'res2.txt')

    alpha_iter = [(10 - x) / 10 for x in range(doc_passes)]
    alpha_iter.reverse()
    model.regularizers['SSThetaMain'].tau = -0.01
    model.regularizers['SSThetaMain'].alpha_iter = alpha_iter
    model.regularizers['SSPhiMain'].tau = -0.01
    model.regularizers.add(artm.ImproveCoherencePhiRegularizer(name='ImpCohPhi',
                                                               tau=1.5,
                                                               class_ids='words',
                                                               dictionary=dictionary))

    model.fit_offline(batch_vectorizer=bv, num_collection_passes=35)
    model.dump_artm_model('250_topics_70itr')

    save_phi_theta(model, save_phi=False)
    get_tokens(model)
    draw_data(model, ConPur=True)
except e:
    raise e
