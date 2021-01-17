import FredMD as fmd


def test_maxfactor():
    # Model should estimate the maximum number of factors as IC chooses 7 normally
    x = fmd.FredMD(Nfactor=None, vintage=None, maxfactor=8, standard_method=2, ic_method=2)
    assert hasattr(x, 'rawseries')
    x.estimate_factors()
    assert x.factors.shape[1] == 7


def test_Nfactor():
    # Model should estimate 2 factors since that is what we are setting
    x = fmd.FredMD(Nfactor=2, vintage=None, maxfactor=1, standard_method=2, ic_method=2)
    x.estimate_factors()
    assert x.factors.shape[1] == 2
