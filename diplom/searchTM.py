import os
import shutil
from time import time
import pandas as pd
from operator import itemgetter
from sklearn.linear_model import LogisticRegression
import numpy as np

def _timeit(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()

        print('{} completed in {:.2f}s'.format(method.__name__, (te-ts)))
        return result
    return timed

class ReadyQuery:
    def __init__(self, model=None):
        """
        :param model: topic model to get query Theta distribution from, if None can't use transform method
        :type model: str or reference to BigARTM artm model object
        """
        if model is not None:
            self._model = self._set_model(model)
        else:
            self._model = None

    @classmethod
    def _set_model(cls, arg_model):
        try:
            import artm
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError('Make sure python knows path to BigARTM library.') from e
        if isinstance(arg_model, str) and os.path.isdir(arg_model):
            model = artm.load_artm_model(arg_model)
        if not isinstance(model, (artm.ARTM, artm.LDA)):
            raise TypeError('Use BigARTM library model')
        return model

    def transform(self, filename):
        """
            :Description: Returns file's Theta distribution which corresponds to model's Theta matrix

            :param str filename: path to the file to get Theta distribution of
            :return:
                * pandas.DataFrame: (data, columns, row), where:
                * columns --- the names of topics in topic model;
                * row --- id of the query document;
                * data --- content of Theta matrix.
            :Note:
                * Needs model to be set
        """
        if self._model is None:
            raise AttributeError('{} object has no model set. Set the model in the constructor'
                                 .format(self.__class__.__name__))
        import artm
        bv_folder = os.path.join(os.getcwd(), 'query_batches')
        query_bv = artm.BatchVectorizer(data_path=filename,
                                        data_format='vowpal_wabbit',
                                        batch_size=10,
                                        target_folder=bv_folder,
                                        gather_dictionary=True)
        query_theta = self._model.transform(batch_vectorizer=query_bv)
        shutil.rmtree(bv_folder)
        return query_theta


class CosineSimilarity(ReadyQuery):
    def __init__(self, theta, qtheta=None, file=None, model=None):
        """
        :param theta: Theta matrix of the topic model
        :type theta: csv or hdf (.h5) table
        :param qtheta: Theta distribution of the query document
        :type qtheta: csv or hdf (.h5) table
        :param str file: path to an unprepared query document
        :param model: topic model to get query Theta distribution from, if None can't get\
                       distribution for a new document
        :type model: str or reference to BigARTM artm model object
        """
        super(CosineSimilarity, self).__init__(model)
        self._theta = self._set_thetas(theta)
        if qtheta is not None:
            self._qtheta = self._set_thetas(qtheta)
        elif file is not None and model is not None:
            super(CosineSimilarity, self).__init__(model=model)
            self._qtheta = self.transform(file)
        else:
            raise ValueError('Either qtheta or file and model must be set')

    @classmethod
    def _set_thetas(cls, theta):
        if isinstance(theta, (pd.DataFrame, pd.Series)):
            if isinstance(theta, pd.Series):
                theta = pd.DataFrame(theta)
            return theta
        elif isinstance(theta, str):
            _, file_extension = os.path.splitext(theta)
            if file_extension == '.csv':
                return pd.read_csv(theta, index_col=0)
            elif file_extension == '.h5':
                return pd.read_hdf(theta, 'table')
            else:
                raise TypeError("Theta matrix must have either hdf (.h5) or csv format. " +
                                "Instead got '{}'".format(file_extension))
        else:
            raise TypeError('theta must be either a DataFrame/Series object or a path to it')

    @property
    def qtheta(self):
        return self._qtheta

    @property
    def theta(self):
        return self._theta

    @qtheta.setter
    def qtheta(self, qtheta):
        self._qtheta = self._set_thetas(qtheta)

    @theta.setter
    def theta(self, theta):
        self._theta = self._set_thetas(theta)

    def get_measure(self, qtheta=None):
        """
            :Description: Gets cosine measure for the specified query document\
                            and documents in the topic model

            :param qtheta: Theta distribution for a new document, None means use one that the object has
            :type qtheta: csv or hdf (.h5) table

            :return: list with tuples of documents ids and their corresponding cosine measure in descending order
        """
        if qtheta is not None:
            self.qtheta = self._set_thetas(qtheta)
        mera = dict()

        mera_down1 = pow(pow(self.qtheta.values[0], 2).sum(), 1/2)
        for row in range(len(self.theta.index)):
            mera_up = self.qtheta.values[0].dot(self.theta.values[row])
            if mera_up == 0:
                continue
            mera_down2 = pow(pow(self.theta.values[row], 2).sum(), 1/2)
            mera[self._theta.index[row]] = mera_up / (mera_down1 * mera_down2)

        mera = sorted(mera.items(), key=itemgetter(1))
        mera.reverse()
        return mera

class QueryAdjust(CosineSimilarity):
    def __init__(self, theta, qtheta=None, file=None, model=None):
        """
           :param theta: Theta matrix of the topic model
           :type theta: csv or hdf (.h5) table
           :param qtheta: Theta distribution of the query document
           :type qtheta: csv or hdf (.h5) table
           :param str file: path to an unprepared query document
           :param model: topic model to get query Theta distribution from, if None can't get\
                          distribution for a new document
           :type model: str or reference to BigARTM artm model object
        """
        super(QueryAdjust, self).__init__(theta, qtheta, file, model)

    def _qurery_adjust(self, rev, irrev, koef):
        nrev = len(rev)
        nirrev = len(irrev)
        n = nrev + nirrev
        if koef is None:
            koef = nrev/nirrev if nirrev > nrev else nirrev/nrev
        q_source = pd.Series(data=self.qtheta.values[0], index=self.qtheta.columns)
        q_rev = pd.Series(data=np.zeros(self.qtheta.shape[1]), index=self.qtheta.columns)
        q_irrev = pd.Series(data=np.zeros(self.qtheta.shape[1]), index=self.qtheta.columns)

        for relev in rev:
            q_rev = q_rev.add(self.theta.loc[relev]).multiply(koef * nirrev / n)
        for irrelev in irrev:
            q_irrev = q_irrev.add(self.theta.loc[irrelev]).multiply(koef * nrev / n)

        res = q_source.add(q_rev).subtract(q_irrev)
        res = (res - res.min()) / (res.max() - res.min())
        res = res.divide(res.sum())
        return pd.DataFrame(res).transpose()

    def adjust(self, rev, irrev, qtheta=None, koef=None):
        """
           :Description: Gets cosine measure for the specified query document\
                          with adjusted to rev and irrev topic distribution and documents in the topic model

           :param list rev: ids of the relevant documents
           :param list rev: ids of the irrelevant documents
           :param qtheta: Theta distribution for a new document, None means use one that the object has
           :type qtheta: csv or hdf (.h5) table

           :return: list with tuples of documents ids and their corresponding cosine measure in descending order
        """
        if qtheta is not None:
            self.qtheta = qtheta                                      # does it call _set_thetas or not?
        mera = self.get_measure(self._qurery_adjust(rev, irrev, koef))
        return mera


class PredictiveAdjust(CosineSimilarity):
    def __init__(self, theta, qtheta=None, file=None, model=None):
        """
             :param theta: Theta matrix of the topic model
             :type theta: csv or hdf (.h5) table
             :param qtheta: Theta distribution of the query document
             :type qtheta: csv or hdf (.h5) table
             :param str file: path to an unprepared query document
             :param model: topic model to get query Theta distribution from, if None can't get\
                            distribution for a new document
             :type model: str or reference to BigARTM artm model object
        """
        super(PredictiveAdjust, self).__init__(theta, qtheta, file, model)

    def _regression_data(self, rev, irrev, mera, confidence):
        if confidence is None:
            confidence = 5e2
        items = len(mera)
        x = np.ones((items, self.theta.shape[1]))
        y = np.zeros(items)
        weights = np.ones(items)
        n = (len(rev) + len(irrev))
        koef_rev = len(irrev) / n
        koef_irrev = len(rev) / n

        for i, item in enumerate(mera):
            x[i] = self.theta.loc[item[0]].values
            if item[0] in rev:
                y[i] = 1
                weights[i] = confidence * koef_rev
            elif item[0] in irrev:
                y[i] = 0
                weights[i] = confidence * koef_irrev
        return x, y, weights

    def _logistic_regression_adjust(self, x, y, weights):
        fitting_index = weights > 1
        logistic = LogisticRegression()
        logistic.fit(x[fitting_index], y[fitting_index], weights[fitting_index])
        return logistic.predict_proba(x)

    def adjust(self, rev, irrev, mera=None, qtheta=None, confidence=None):
        """
           :Description: Gets logistic regression model's prediction regarding relevance of the\
                         document in topic model to the query document

           :param list rev: ids of the relevant documents
           :param list rev: ids of the irrelevant documents
           :param mera: cosine measure of the document to predict relevance of, None means\
                         predict for all document in the topic model
           :type mera: list of tuples with documents ids and their corresponding cosine measure
           :param qtheta: Theta distribution for a new document, None means use one that the object has
           :type qtheta: csv or hdf (.h5) table

           :return:
            *list with triple tuples with:
            *document ids;
            *corresponding cosine measure;
            *corresponding relevance prediction (from 0 to 1);
            sorted by relevance in descending order
        """
        if mera is None:
            mera = self.get_measure(qtheta)
        x, y, weights = self._regression_data(rev, irrev, mera, confidence)
        predicted = self._logistic_regression_adjust(x, y, weights)
        predicted = [(*x,y[1]) for (x, y) in sorted(zip(mera, predicted),
                                            key=lambda pair: pair[1][1], reverse=True)]
        return predicted
