import numpy as np
from scipy.stats import norm
import pandas as pd
Z = norm.ppf


def d_prime(tpr, fpr):
    return Z(tpr) - Z(fpr)


def c_bias(tpr, fpr):
    return 1 / 2 * -(Z(tpr) + Z(fpr))


def A_prime(tpr, fpr):

    if tpr >= fpr:
        numerator = ((tpr - fpr) * (1 + tpr - fpr))
        denominator = 4 * tpr * (1 - fpr)
        return 0.5 + (numerator / denominator)
    else:
        numerator = ((fpr - tpr) * (1 + fpr - tpr))
        denominator = 4 * fpr * (1 - tpr)
        return 0.5 - (numerator / denominator)


def beta(tpr, fpr):
    return np.exp((1 / 2) * Z(fpr)**2 - Z(tpr))**2


def beta_doubleprime(tpr, fpr, donaldson=False):
    """Default is Grier's (1971) as cited in Stanislaw & Todorov (1999).
    """
    fnr = 1 - tpr
    tnr = 1 - fpr

    if donaldson:
        numerator = (fnr * tnr) - (tpr * fpr)
        denominator = (fnr * tnr) + (tpr * fpr)
        return numerator / denominator

    else:
        numerator = (tpr * fnr) - (fpr * tnr)
        denominator = (tpr * fnr) + (fpr * tnr)
        return np.sign(tpr - fpr) * numerator / denominator


def true_positive_rate(y_true, y_resp, norm=False, corrected=False):
    true_positives = np.nansum(y_resp[y_true == 1])

    if not norm:
        # Just return the total, no need to correct.
        return true_positives

    n_positive = len(y_true[y_true == 1])

    if not corrected:
        return true_positives / n_positive
    else:
        return (true_positives + 0.5) / (n_positive + 1)


def false_negative_rate(y_true, y_resp, norm=False, corrected=False):
    false_negatives = np.nansum(np.abs(y_resp - 1)[y_true == 1])

    if not norm:
        return false_negatives

    n_positive = len(y_true[y_true == 1])

    if not corrected:
        return false_negatives / n_positive
    else:
        return (false_negatives + 0.5) / (n_positive + 1)


def false_positive_rate(y_true, y_resp, norm=False, corrected=False):
    false_positives = np.nansum(y_resp[y_true == 0])

    if not norm:
        return false_positives
    n_negative = len(y_true[y_true == 0])

    if not corrected:
        return false_positives / n_negative
    else:
        return (false_positives + 0.5) / (n_negative + 1)


def true_negative_rate(y_true, y_resp, norm=False, corrected=False):
    true_negatives = np.nansum(np.abs(y_resp - 1)[y_true == 0])

    if not norm:
        return true_negatives

    n_negative = len(y_true[y_true == 0])

    if not corrected:
        return true_negatives / n_negative
    else:
        return (true_negatives + 0.5) / (n_negative + 1)


def confusion_matrix(y_true, y_resp, norm=False, corrected=False):
    '''Returns a confusion matrix from the vectors y_true and y_resp.'''
    tpr = true_positive_rate(y_true, y_resp, norm=norm, corrected=corrected)
    fpr = false_positive_rate(y_true, y_resp, norm=norm, corrected=corrected)
    fnr = false_negative_rate(y_true, y_resp, norm=norm, corrected=corrected)
    tnr = true_negative_rate(y_true, y_resp, norm=norm, corrected=corrected)

    cm = pd.DataFrame(
        {'positive': [tpr, fpr], 'negative': [tnr, fnr]},
        index=['true', 'false'])

    return cm


def compute(y_true, y_resp, group=None, data=None, corrected=True):
    '''Computes signal detection measures d` and C from data given.'''
    if data is None:
        # TODO: allow non dataframe
        # y_true & y_resp & group must be array-like
        # create a pandas dataframe

        pass

    if group is not None:
        data = data.groupby(group)
    else:
        data[''] = 'result'
        data = data.groupby('')

    results = []

    funcs = {'tpr': true_positive_rate, 'fpr': false_positive_rate}

    for funcname, func in funcs.items():
        result = data.apply(lambda x: func(x[y_true], x[y_resp],
                                           norm=True, corrected=corrected))
        result.name = funcname
        results.append(result)

    results = pd.concat(results, axis=1)

    results['d_prime'] = results.apply(
        lambda x: d_prime(x['tpr'], x['fpr']), axis=1)

    results['A_prime'] = results.apply(
        lambda x: A_prime(x['tpr'], x['fpr']), axis=1)

    results['c_bias'] = results.apply(
        lambda x: c_bias(x['tpr'], x['fpr']), axis=1)

    results['B'] = results.apply(lambda x: beta(x['tpr'], x['fpr']), axis=1)

    results['B_doubleprime_D'] = \
        results.apply(
            lambda x: beta_doubleprime(x['tpr'], x['fpr'], donaldson=True),
            axis=1)

    return results
