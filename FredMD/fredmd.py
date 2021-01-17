import pandas as pd
import numpy as np
import sklearn.decomposition as skd
import sklearn.preprocessing as skp
import sklearn.pipeline as skpipe
import math


class FredMD:
    """
    FredMD object. Creates factors based off the FRED-MD dataset (https://research.stlouisfed.org/econ/mccracken/fred-databases/)
    Methods:
    1) FredMD(): initialize object with downloaded data
    2) estimate_factors(): Runs the full estimation
    3) factors_em(): Estimates factors with the EM alogrithm to handle missing observations
    4) baing(): Estimates the Bai-Ng factor selection alogrithm
    5) apply_transforms(): Apply the transform to each series
    6) remove_outliers(): Removes Outliers
    7) factor_standardizer_method(): Converts standard_method to appropiate sklearn.StandardScaler
    8) data_transforms(): Applies function to series to make data stationary as given by transform code
    9) download_data(): Download FRED-MD dataset
    10) V(): Explained variance function
    """

    def __init__(self, Nfactor=None, vintage=None, maxfactor=8, standard_method=2, ic_method=2) -> None:
        """
        Create fredmd object
        Auguments:
        1) Nfactor = None: Number of factors to estimate. If None then estimate number of true factors via information critea
        2) vintage = None: Vinatege of data to use in "year-month" format (e.g. "2020-10"). If None use current vintage
        3) maxfactor = 8: Maximimum number of factors to test against information critea. If Nfactor is a number, then this is ignored
        4) standard_method = 2: method to standardize data before factors are estimate. 0 = Identity transform, 1 = Demean only, 2 = Demean and stardize to unit variance. Default = 2.
        5) ic_method = 2: information critea penalty term. See http://www.columbia.edu/~sn2294/pub/ecta02.pdf page 201, equation 9 for options.
        """
        # Make sure arguments are valid
        if standard_method not in [0, 1, 2]:
            raise ValueError(
                f"standard_method must be in [0, 1, 2], got {standard_method}")
        if ic_method not in [1, 2, 3]:
            raise ValueError(
                f"ic_method must be in [1, 2, 3], got {ic_method}")

        # Download data
        self.rawseries, self.transforms = self.download_data(vintage)
        # Check maxfactor
        if maxfactor > self.rawseries.shape[1]:
            raise ValueError(
                f"maxfactor must be less then number of series. Maxfactor({maxfactor}) > N Series({self.rawseries.shape[1]})")

        self.standard_method = standard_method
        self.ic_method = ic_method
        self.maxfactor = maxfactor
        self.Nfactor = Nfactor

    @staticmethod
    def download_data(vintage):
        if vintage is None:
            url = 'https://s3.amazonaws.com/files.fred.stlouisfed.org/fred-md/monthly/current.csv'
        else:
            url = f'https://s3.amazonaws.com/files.fred.stlouisfed.org/fred-md/monthly/{vintage}.csv'
        print(url)
        transforms = pd.read_csv(
            url, header=0, nrows=1, index_col=0).transpose()
        transforms.index.rename("series", inplace=True)
        transforms.columns = ['transform']
        transforms = transforms.to_dict()['transform']
        data = pd.read_csv(url, names=transforms.keys(), skiprows=2, index_col=0,
                           skipfooter=1, engine='python', parse_dates=True, infer_datetime_format=True)
        return data, transforms

    @staticmethod
    def factor_standardizer_method(code):
        """
        Outputs the sklearn standard scaler object with the desired features
        codes:
        0) Identity transform
        1) Demean only
        2) Demean and standardized
        """
        if code == 0:
            return skp.StandardScaler(with_mean=False, with_std=False)
        elif code == 1:
            return skp.StandardScaler(with_mean=True, with_std=False)
        elif code == 2:
            return skp.StandardScaler(with_mean=True, with_std=True)
        else:
            raise ValueError("standard_method must be in [0, 1, 2]")

    @staticmethod
    def data_transforms(series, transform):
        """
        Transforms a single series according to its transformation code
        Inputs:
        1) series: pandas series to be transformed
        2) transfom: transform code for the series
        Returns:
        transformed series
        """
        if transform == 1:
            # level
            return series
        elif transform == 2:
            # 1st difference
            return series.diff()
        elif transform == 3:
            # second difference
            return series.diff().diff()
        elif transform == 4:
            # Natural log
            return np.log(series)
        elif transform == 5:
            # log 1st difference
            return np.log(series).diff()
        elif transform == 6:
            # log second difference
            return np.log(series).diff().diff()
        elif transform == 7:
            # First difference of percent change
            return series.pct_change().diff()
        else:
            raise ValueError("Transform must be in [1, 2, ..., 7]")

    def apply_transforms(self):
        """
        Apply the transformation to each series to make them stationary and drop the first 2 rows that are mostly NaNs
        Save results to self.series
        """
        self.series = pd.DataFrame({key: self.data_transforms(
            self.rawseries[key], value) for (key, value) in self.transforms.items()})
        self.series.drop(self.series.index[[0, 1]], inplace=True)

    def remove_outliers(self):
        """
        Removes outliers from each series in self.series
        Outlier definition: a data point x of a series X is considered an outlier if abs(x-median)>10*interquartile_range.
        """
        Z = abs((self.series - self.series.median()) /
                (self.series.quantile(0.75) - self.series.quantile(0.25))) > 10
        for col, _ in self.series.iteritems():
            self.series[col][Z[col]] = np.nan

    def factors_em(self, max_iter=50, tol=math.sqrt(0.000001)):
        """
        Estimates factors with EM alogorithm to handle missings
        Inputs:
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence between iterations of predicted series values
        Alogrithm:
        1) initial_nas: Boolean mask of locations of NaNs
        2) working_data: Create Standardized data matrix with nan's replaced with means
        3) F: Preliminary factor estimates
        4) data_hat_last: Predicted standardized values of last SVD model. data_hat and data_hat_last will not exactly be mean 0 variance 1
        5) Iterate data_hat until convergence
        6) Fill in nans from orginal data
        Saves
        1) self.svdmodel: sklearn pipeline with standardization step and svd model
        2) self.series_filled: self.series with any NaNs filled in with predicted values from self.svdmodel
        """
        # Define our estimation pipelines
        pipe = skpipe.Pipeline([('Standardize', self.factor_standardizer_method(
            self.standard_method)), ('Factors', skd.TruncatedSVD(self.Nfactor, algorithm='arpack'))])
        inital_scalar = self.factor_standardizer_method(self.standard_method)

        # Make numpy arrays for calculations
        actual_data = self.series.to_numpy(copy=True)
        intial_nas = self.series.isna().to_numpy(copy=True)
        working_data = inital_scalar.fit_transform(self.series.fillna(
            value=self.series.mean(), axis='index').to_numpy(copy=True))

        # Estimate initial model
        F = pipe.fit_transform(working_data)
        data_hat_last = pipe.inverse_transform(F)

        # Iterate until model convereges
        iter = 0
        distance = tol+1
        while (iter < max_iter) and (distance > tol):
            F = pipe.fit_transform(working_data)
            data_hat = pipe.inverse_transform(F)
            distance = np.linalg.norm(
                data_hat-data_hat_last, 2)/np.linalg.norm(data_hat_last, 2)
            data_hat_last = data_hat.copy()
            working_data[intial_nas] = data_hat[intial_nas]
            iter += 1

        # Print results
        if iter == max_iter:
            print(
                f"EM alogrithm failed to converge afet Maximum iterations of {max_iter}. Distance = {distance}, tolerance was {tol}")
        else:
            print(f"EM algorithm converged after {iter} iterations")

        # Save Results
        actual_data[intial_nas] = inital_scalar.inverse_transform(working_data)[
            intial_nas]
        self.svdmodel = pipe
        self.series_filled = pd.DataFrame(
            actual_data, index=self.series.index, columns=self.series.columns)
        self.factors = pd.DataFrame(F, index=self.series_filled.index, columns=[
                                    f"F{i}" for i in range(1, F.shape[1]+1)])

    @staticmethod
    def V(X, F, Lambda):
        """
        Explained Variance of X by factors F with loadings Lambda
        """
        T, N = X.shape
        NT = N*T
        return np.linalg.norm(X - F @ Lambda, 2)/NT

    def baing(self):
        """
        Determine the number of factors to use using the Bai-Ng Information Critrion
        reference: http://www.columbia.edu/~sn2294/pub/ecta02.pdf
        """
        # Define our estimation pipelines
        pipe = skpipe.Pipeline([('Standardize', self.factor_standardizer_method(
            self.standard_method)), ('Factors', skd.TruncatedSVD(self.maxfactor, algorithm='arpack'))])
        inital_scalar = self.factor_standardizer_method(self.standard_method)

        # Setup
        working_data = inital_scalar.fit_transform(self.series.fillna(
            value=self.series.mean(), axis='index').to_numpy(copy=True))
        T, N = working_data.shape
        NT = N*T
        NT1 = N+T
        # Make information critea penalties
        if self.ic_method == 1:
            CT = [i * math.log(NT/NT1) * NT1/NT for i in range(self.maxfactor)]
        elif self.ic_method == 2:
            CT = [i * math.log(min(N, T)) * NT1 /
                  NT for i in range(self.maxfactor)]
        elif self.ic_method == 3:
            CT = [i * math.log(min(N, T)) / min(N, T)
                  for i in range(self.maxfactor)]
        else:
            raise ValueError("ic must be either 1, 2 or 3")

        # Fit model with max factors
        F = pipe.fit_transform(working_data)
        Lambda = pipe['Factors'].components_
        Vhat = [self.V(working_data, F[:, 0:i], Lambda[0:i, :])
                for i in range(self.maxfactor)]
        IC = np.log(Vhat) + CT
        kstar = np.argmin(IC)
        self.Nfactor = kstar

    def estimate_factors(self):
        """
        Runs estimation routine.
        If number of factors is not specified then estimate the number to be used
        """
        self.apply_transforms()
        self.remove_outliers()
        if self.Nfactor is None:
            self.baing()
        self.factors_em()
