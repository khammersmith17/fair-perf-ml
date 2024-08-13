from typing import Union
import numpy as np


class PostTrainingBias:
    @staticmethod
    def ddpl(facet_a_scores: np.ndarray, facet_d_scores: np.ndarry, facet_a_trues: np.ndarray, facet_d_trues: np.ndarray) -> float:
        q_prime_a = np.sum(facet_a_scores)/np.sum(facet_a_trues)
        q_prime_d = np.sum(facet_d_scores)/np.sum(facet_d_trues)
        return q_prime_a - q_prime_d

    @staticmethod
    def di(facet_a_scores: np.ndarray, facet_d_scores: np.ndarry, facet_a_trues: np.ndarray, facet_d_trues: np.ndarray) -> float:
        q_prime_a = np.sum(facet_a_scores)/np.sum(facet_a_trues)
        q_prime_d = np.sum(facet_d_scores)/np.sum(facet_d_trues)
        return q_prime_a / q_prime_d

    @staticmethod
    def ad(facet_a_scores: np.ndarray, facet_d_scores: np.ndarry, facet_a_trues: np.ndarray, facet_d_trues: np.ndarray) -> float:
        tp_a = np.sum(np.where(np.logical_and(facet_a_scores == 1, facet_a_trues == 1), 1, 0))
        tp_d = np.sum(np.where(np.logical_and(facet_d_scores == 1, facet_d_trues == 1), 1, 0))
        tn_a = np.sum(np.where(np.logical_and(facet_a_scores == 0, facet_a_trues == 0), 1, 0))
        tn_d = np.sum(np.where(np.logical_and(facet_d_scores == 0, facet_d_trues == 0), 1, 0))

        acc_a = (tp_a + tn_a) / facet_a_scores.shape[0]
        acc_d = (tp_d + tn_d) / facet_d_scores.shape[0]

        return acc_a - acc_d

    @staticmethod
    def rd(facet_a_scores: np.ndarray, facet_d_scores: np.ndarry, facet_a_trues: np.ndarray, facet_d_trues: np.ndarray) -> Union[float, None]:
        tp_a = np.sum(np.where(np.logical_and(facet_a_scores == 1, facet_a_trues == 1), 1, 0))
        fn_a = np.sum(np.where(np.logical_and(facet_a_scores == 0, facet_a_trues == 1), 1, 0))
        if fn_a == 0.0:
            return None

        tp_d = np.sum(np.where(np.logical_and(facet_d_scores == 1, facet_d_trues == 1), 1, 0))
        fn_d = np.sum(np.where(np.logical_and(facet_d_scores == 0, facet_d_trues == 1), 1, 0))

        if fn_d == 0.0:
            return None

        recall_a = tp_a / (tp_a + fn_a)
        recall_d = tp_d / (tp_d + fn_d)
        return recall_a - recall_d

    @staticmethod
    def cdacc(facet_a_scores: np.ndarray, facet_d_scores: np.ndarry, facet_a_trues: np.ndarray, facet_d_trues: np.ndarray) -> float:
        c_a = np.sum(facet_a_trues) / np.sum(facet_a_scores)
        c_d = np.sum(facet_d_trues) / np.sum(facet_d_scores)
        return c_a - c_d

    @staticmethod
    def dar(facet_a_scores: np.ndarray, facet_d_scores: np.ndarry, facet_a_trues: np.ndarray, facet_d_trues: np.ndarray) -> Union[float, None]:
        tp_a = np.sum(np.where(np.logical_and(facet_a_scores == 1, facet_a_trues == 1), 1, 0))
        fp_a = np.sum(np.where(np.logical_and(facet_a_scores == 1, facet_a_trues == 0), 1, 0))

        if fp_a == 0.0:
            return None

        tp_d = np.sum(np.where(np.logical_and(facet_d_scores == 1, facet_d_trues == 1), 1, 0))
        fp_d = np.sum(np.where(np.logical_and(facet_d_scores == 1, facet_d_trues == 0), 1, 0))

        if fp_d == 0.0:
            return None

        precision_a = tp_a + (tp_a + fp_a)
        precision_d = tp_d + (tp_d + fp_d)

        return precision_a - precision_d

    @staticmethod
    def sd(facet_a_scores: np.ndarray, facet_d_scores: np.ndarry, facet_a_trues: np.ndarray, facet_d_trues: np.ndarray) -> Union[float, None]:
        tn_d = np.sum(np.where(np.logical_and(facet_d_trues == 0, facet_d_scores == 0), 1, 0))
        fp_d = np.sum(np.where(np.logical_and(facet_d_trues == 1, facet_d_scores == 0), 1, 0))
        if fp_d == 0.0:
            return None

        tn_a = np.sum(np.where(np.logical_and(facet_a_trues == 0, facet_a_scores == 0), 1, 0))
        fp_a = np.sum(np.where(np.logical_and(facet_a_trues == 1, facet_a_scores == 0), 1, 0))
        if fp_a == 0.0:
            return None

        tnr_d = tn_d / (tn_d + fp_d)
        tnr_a = tn_a / (tn_a + fp_a)

        return tnr_d - tnr_a

    @staticmethod
    def dcr(facet_a_scores: np.ndarray, facet_d_scores: np.ndarry, facet_a_trues: np.ndarray, facet_d_trues: np.ndarray) -> float:
        n_prime_d = np.sum(np.where(facet_d_scores == 0, 1, 0))
        n_d = np.sum(np.where(facet_d_trues == 0, 1, 0))
        r_d = n_d/n_prime_d

        n_prime_a = np.sum(np.where(facet_a_scores == 0, 1, 0))
        n_a = np.sum(np.where(facet_a_trues == 1, 1, 0))
        r_a = n_a/n_prime_a

        return r_d - r_a

    @staticmethod
    def drr(facet_a_scores: np.ndarray, facet_d_scores: np.ndarry, facet_a_trues: np.ndarray, facet_d_trues: np.ndarray) -> float:
        tn_d = np.sum(np.where(np.logical_and(facet_d_scores == 0, facet_d_trues == 0), 1, 0))
        fn_d = np.sum(np.where(np.logical_and(facet_d_scores == 0, facet_d_trues == 1), 1, 0))
        val_d = tn_d / (tn_d + fn_d)

        tn_a = np.sum(np.where(np.logical_and(facet_a_scores == 0, facet_a_trues == 0), 1, 0))
        fn_a = np.sum(np.where(np.logical_and(facet_a_scores == 0, facet_a_trues == 1), 1, 0))
        val_a = tn_a / (tn_a + fn_a)

        return val_d - val_a

    @staticmethod
    def te(facet_a_scores: np.ndarray, facet_d_scores: np.ndarry, facet_a_trues: np.ndarray, facet_d_trues: np.ndarray) -> float:
        fn_d = np.sum(np.where(np.logical_and(facet_d_scores == 0, facet_d_trues == 1), 1, 0))
        fp_d = np.sum(np.where(np.logical_and(facet_d_scores == 1, facet_d_trues == 0), 1, 0))
        val_d = fn_d / fp_d

        fn_a = np.sum(np.where(np.logical_and(facet_a_scores == 0, facet_a_trues == 1), 1, 0))
        fp_a = np.sum(mp.where(np.logical_and(facet_a_scores == 1, facet_a_trues == 0), 1, 0))
        val_a = fn_a / fp_a

        return val_d - val_a

    @staticmethod
    def ccdpl(facet_a_scores: np.ndarray, facet_d_scores: np.ndarray) -> float:
        n_prime_0 = np.sum(np.where(facet_a_scores == 0, 1, 0)) + np.sum(np.where(facet_d_scores == 0, 1, 0))
        n_prime_1 = np.sum(np.where(facet_a_scores == 1, 1, 0)) + np.sum(np.where(facet_d_scores == 1, 1, 0))
        n_prime_d_0 = np.sum(np.where(facet_d_scores == 0, 1, 0))
        n_prime_d_1 = np.sum(np.where(facet_d_scores == 1, 1, 0))

        return (n_prime_d_0/n_prime_0) - (n_prime_d_1/n_prime_1)

    @staticmethod
    def ge(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        benefits = []
        for pred, true in zip(y_pred, y_true):
            if pred == 0 and true == 1:
                benefits.append(0)
            elif pred == 1 and true == 1:
                benefits.append(1)
            elif pred == 1 and true == 0:
                benefits.append(2)
        benefits = np.array(benefits)
        mean = benefits.mean()
        benefits = ((benefits/mean) ** 2) - 1
        n = benefits.shape[0]
        return np.sum(benefits) * (1/2*n)
