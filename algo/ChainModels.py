import numpy as np
import pandas as pd
from scipy.stats import linregress
from random import shuffle
import statsmodels.api as sm




def simulate_GLM(model_result,df) :
    df['year'] = pd.Categorical(df['year'])
    df['dev']  = pd.Categorical(df['dev'] )
    X_custom = pd.get_dummies(df[['year', 'dev']], drop_first=True)
    if 'const' not in model_result.params.index:
        X_custom = sm.add_constant(X_custom, has_constant='add') # Create dummy variables
    linear_predictor = np.dot(X_custom, model_result.params)
    linear_predictor = np.array(linear_predictor,dtype=float)
    link_function    = model_result.family.link.inverse
    mu = link_function(linear_predictor)
    if isinstance(model_result.family, sm.families.Gaussian):
        y = mu + np.random.normal(scale=model_result.scale, size=X_custom.shape[0])
    elif isinstance(model_result.family, sm.families.Poisson):
        y = np.random.poisson(lam=mu, size=X_custom.shape[0])
    elif isinstance(model_result.family, sm.families.Binomial):
        y = np.random.binomial(1, p=mu, size=X_custom.shape[0])
    else:
        raise ValueError("Unsupported family: Only Gaussian, Poisson, and Binomial are implemented.")
    simulated_data = pd.DataFrame(X_custom)
    simulated_data['y'] = y
    return simulated_data['y']

class Triangle:
    def __init__(self, years, data, isCumul=True, inferior_nan = False):
        """
        Initialize a Triangle object.

        Args:
            years (list): List of years.
            data (numpy.ndarray): Triangle data (cumulative or incremental).
            isCumul (bool, optional): Whether the data is cumulative. Defaults to True.
        """
        self.years = years
        if isCumul:
            dataCumul = data
            dataIncrement = np.concatenate((data[:, 0].reshape(-1, 1), np.diff(data, axis=1)), axis=1)
        else:
            dataCumul = data.cumsum(axis=1)
            dataIncrement = data
        if inferior_nan == True : 
            for i in range(data.shape[0]) : 
                for j in range(data.shape[1]-i,data.shape[1]) : 
                    dataIncrement[i,j]  = np.nan 
                    dataCumul[i,j]      = np.nan

        self.Inc = dataIncrement
        self.Cum = dataCumul
        self.shape = self.Cum.shape

    def extract(self, start, end):
        """
        Extract a subset of data based on start and end years.

        Args:
            start (int): Start year.
            end (int): End year.

        Returns:
            Triangle: New Triangle object with extracted data.
        """
        first_index = None
        last_index = None

        for i, year in enumerate(self.years):
            if year >= start and first_index is None:
                first_index = i
            if year <= end:
                last_index = i

        newyears = self.years[first_index:last_index + 1]
        newdata = self.Cum[first_index:last_index + 1, first_index:last_index + 1]
        return Triangle(years=newyears, data=newdata, isCumul=True)

    def __str__(self):
        """
        Return a string representation of the Triangle object.

        Returns:
            str: String representation.
        """
        df = pd.DataFrame(self.Inc, index=self.years, columns=range(self.shape[1]))
        dfc = pd.DataFrame(self.Cum, index=self.years, columns=range(self.shape[1]))
        return "Increment = \n" + df.__str__() + "\nCumuls = \n" + dfc.__str__()

class ChainLadder:
    """
    ChainLadder class for performing chain ladder analysis on triangle data.

    Attributes:
        DevFactors (numpy.ndarray): Development factors calculated during the fitting process.
        FullTriangle (Triangle): Triangle object containing the full development data after fitting.
    """
    def __init__(self):
        """
        Initializes an instance of the ChainLadder class.
        """
        self.DevFactors = None
        self.FullTriangle = None

    def fit(self, triangle: Triangle):
        """
        Fits the ChainLadder model to the provided triangle data.

        Args:
            triangle (Triangle): Triangle object containing the development data.

        Returns:
            None
        """

        n_row, n_col = triangle.Cum.shape
        DevFact = np.zeros(n_col - 1)
        for j in range(n_col - 1):
            DevFact[j] = np.sum(triangle.Cum[:n_row - j - 1, j + 1]) / np.sum(triangle.Cum[:n_row - j - 1, j])
        FullTriangle = triangle.Cum.copy()
        for j in range(n_col - 1):
            FullTriangle[n_row - j - 1:, j + 1] = FullTriangle[n_row - j - 1:, j] * DevFact[j]
        self.DevFactors = DevFact
        self.FullTriangle = Triangle(years=triangle.years, data=FullTriangle, isCumul=True)

    def BackEstimates(self) :
        data  = self.FullTriangle.Cum.copy() 
        n_row,n_col = data.shape
        for i in range(n_row):
            for j in range(n_row-i-1, 0, -1):  
                data[i, j-1] = data[i, j] / self.DevFactors[j-1]
        return Triangle(years=self.FullTriangle.years,data=data,isCumul=True)

    def Residuals(self) : 
        data  = self.FullTriangle.Cum.copy() 
        n_row,n_col = data.shape
        residual = (self.BackEstimates() - data)/np.sqrt(data)
        return residual

    def Simulate(self) : 
        resids = self.Residuals()
        flattened = resids.flatten()
        np.random.shuffle(flattened)
        resids = flattened.reshape(resids.shape)
        simulation = self.BackEstimates().Inc + np.sqrt(resids)*self.BackEstimates().Inc
        return Triangle(years=self.FullTriangle.years,data=simulation,isCumul=True)

    def Provisions(self):
        """
        Calculate provisions of Chain Ladder model for each year based on the full triangle data.

        Returns:
            pandas.DataFrame: DataFrame containing the year and corresponding provisions.
        """

        FullTriangle = self.FullTriangle.Cum
        n_row, n_col = FullTriangle.shape
        prov = np.array([
            FullTriangle[i, -1] - FullTriangle[i, n_row - i - 1] for i in range(n_row)
        ])
        return pd.DataFrame({
            "Year": self.FullTriangle.years,
            "Provision": prov
        })

    def __str__(self):
        """
        Return a string representation of the Chain Ladder Model object.

        Returns:
            str: String representation of the object.
        """
        return f"This is a Chain Ladder Model, with development factors: {self.DevFactors}\nAnd Full triangle estimated:\n{self.FullTriangle}"

class ChainLondon:
    """
    ChainLondon class for performing London Chain Ladder analysis on triangle data.

    Attributes:
        Slopes (numpy.ndarray): Slope factors calculated during the fitting process.
        Intercepts (numpy.ndarray): Intercept factors calculated during the fitting process.
        FullTriangle (Triangle): Triangle object containing the full development data after fitting.
    """

    def __init__(self):
        """
        Initializes an instance of the ChainLondon class.
        """
        self.Slopes = None
        self.Intercepts = None
        self.FullTriangle = None

    def fit(self, triangle: Triangle):
        """
        Fits the ChainLondon model to the provided triangle data.

        Args:
            triangle (Triangle): Triangle object containing the development data.

        Returns:
            None
        """

        n_row, n_col = triangle.Cum.shape
        Slopes = np.zeros(n_col - 1)
        Intercepts = np.zeros(n_col - 1)

        for j in range(n_col - 1):
            x = triangle.Cum[:n_row - j - 1, j]
            y = triangle.Cum[:n_row - j - 1, j + 1]
            
            if len(x) == 1:
                Slopes[j], Intercepts[j] = y[0] / x[0], 0
            else:
                # Perform linear regression
                Slopes[j], Intercepts[j], _, _, _ = linregress(x, y)

        self.Slopes = Slopes
        self.Intercepts = Intercepts

        FullTriangle = triangle.Cum.copy()
        for j in range(n_col - 1):
            FullTriangle[n_row - j - 1:, j + 1] = FullTriangle[n_row - j - 1:, j] * Slopes[j] + Intercepts[j]

        self.FullTriangle = Triangle(years=triangle.years, data=FullTriangle, isCumul=True)

    def Provisions(self):
        """
        Calculate provisions for each year based on the full triangle data using London Chain Ladder method.

        Returns:
            pandas.DataFrame: DataFrame containing the year and corresponding provisions.
        """

        FullTriangle = self.FullTriangle.Cum
        n_row, n_col = FullTriangle.shape
        prov = np.array([
            FullTriangle[i, -1] - FullTriangle[i, n_row - i - 1] for i in range(n_row)
        ])
        return pd.DataFrame({
            "Year": self.FullTriangle.years,
            "Provision": prov
        })

    def __str__(self):
        """
        Return a string representation of the Chain London Model object.

        Returns:
            str: String representation of the object.
        """
        return f"This is a Chain London Model, with Slope factors: {self.Slopes},\nIntercept factors: {self.Intercepts}\nAnd Full triangle estimated:\n{self.FullTriangle}"

class ChainMack:
    """
    ChainMack class for performing Mack Chain Ladder analysis on triangle data.

    Attributes:
        DevFactors (numpy.ndarray): Development factors calculated during the fitting process.
        Deviations (numpy.ndarray): Deviation factors calculated during the fitting process.
        FullTriangle (Triangle): Triangle object containing the full development data after fitting.
    """

    def __init__(self):
        """
        Initializes an instance of the ChainMack class.
        """
        self.DevFactors = None
        self.Deviations = None
        self.FullTriangle = None

    def fit(self, triangle: Triangle):
        """
        Fits the ChainMack model to the provided triangle data.
        For more mathematical background see : [https://actuaries.org/LIBRARY/ASTIN/vol23no2/213.pdf]

        Args:
            triangle (Triangle): Triangle object containing the development data.

        Returns:
            None
        """

        n_row, n_col = triangle.Cum.shape
        DevFact = np.zeros(n_col - 1)
        Sigmas  = np.zeros(n_col - 1)
        for j in range(n_col - 1):
            DevFact[j] = np.sum(triangle.Cum[:n_row - j - 1, j + 1]) / np.sum(triangle.Cum[:n_row - j - 1, j])
            if j < n_col - 1 - 1:
                Sigmas[j] = np.sqrt(1 / (n_row - j - 2 )*np.sum( triangle.Cum[:n_row - j - 1, j]*(triangle.Cum[:n_row - j - 1, j + 1]/triangle.Cum[:n_row - j - 1, j] - DevFact[j] )**2 ))
                
        Sigmas[n_col - 1 - 1] = min(min(Sigmas[n_col - 1 - 2], Sigmas[n_col - 1 - 3]),
                                     Sigmas[n_col - 1 - 2] ** 2 / Sigmas[n_col - 1 - 3])

        FullTriangle = triangle.Cum.copy()
        for i in range(n_row - 1, n_col - n_row - 1, -1):
            FullTriangle[i, n_row - i:] = FullTriangle[i, n_row - i - 1] * np.cumprod(DevFact[n_row - i - 1:])

        self.DevFactors = DevFact
        self.Deviations = Sigmas
        self.FullTriangle = Triangle(years=triangle.years, data=FullTriangle, isCumul=True)

    def refit(self, triangle:Triangle) : 
        n_row, n_col = triangle.Cum.shape
        DevFact = np.zeros(n_col - 1)
        Sigmas  = np.zeros(n_col - 1)
        for j in range(n_col - 1):
            DevFact[j] = np.sum(triangle.Cum[:n_row - j - 1, j + 1]) / np.sum(triangle.Cum[:n_row - j - 1, j])
            if j < n_col - 1 - 1:
                Sigmas[j] = np.sqrt(1 / (n_row - j - 2 )*np.sum( triangle.Cum[:n_row - j - 1, j]*(triangle.Cum[:n_row - j - 1, j + 1]/triangle.Cum[:n_row - j - 1, j] - DevFact[j] )**2 ))
                
        Sigmas[n_col - 1 - 1] = min(min(Sigmas[n_col - 1 - 2], Sigmas[n_col - 1 - 3]),
                                     Sigmas[n_col - 1 - 2] ** 2 / Sigmas[n_col - 1 - 3])

        FullTriangle = triangle.Cum.copy()
        for i in range(n_row - 1, n_col - n_row - 1, -1):
            FullTriangle[i, n_row - i:] = FullTriangle[i, n_row - i - 1] * np.cumprod(DevFact[n_row - i - 1:])

        return Triangle(years=triangle.years, data=FullTriangle, isCumul=True)

    def BackEstimates(self) :
        data  = self.FullTriangle.Cum.copy() 
        n_row,n_col = data.shape
        for i in range(n_row):
            for j in range(n_row-i-1, 0, -1):  
                data[i, j-1] = data[i, j] / self.DevFactors[j-1]
        return Triangle(years=self.FullTriangle.years,data=data,isCumul=True)

    def Residuals(self) : 
        data  = self.FullTriangle.Inc.copy() 
        residual = (self.BackEstimates().Inc - data)/np.sqrt(data)
        return residual

    def Simulate(self,method = "Bootstrap") : 
        if method == "Bootstrap" :
            resids = self.Residuals()
            non_null_elements = resids[resids != 0]
            np.random.shuffle(non_null_elements)
            shuffled_matrix = resids.copy()
            shuffled_matrix[resids != 0] = non_null_elements
            simulation = self.BackEstimates().Inc + np.sqrt(self.BackEstimates().Inc)*shuffled_matrix
            return self.refit(Triangle(years=self.FullTriangle.years,data=simulation,isCumul=False))
        else  : 
            FullTriangle = self.FullTriangle.Cum.copy()
            n_row, n_col = FullTriangle.shape
            mses = self.Provisions()['MSE']
            hatR = np.array([ FullTriangle[i, -1]  for i in range(n_row) ])
            sigma = np.sqrt( np.log(1+ (mses/hatR)**2 ) )
            mu    = np.log( hatR ) - sigma**2/2 
            FullTriangle[:,-1] = np.exp( np.random.normal(mu,sigma) )  if np.any(sigma != 0) else mu
            return Triangle(years=self.FullTriangle.years,data=FullTriangle,isCumul=True)

    def Provisions(self):
        """
        Calculate the provisions for each year based on the fitted Mack Chain Ladder model.

        Returns:
            pandas.DataFrame: DataFrame containing the years, provisions, and mean squared errors (MSE).
        """

        FullTriangle = self.FullTriangle.Cum
        n_row, n_col = FullTriangle.shape
        prov = np.array([
            FullTriangle[i, -1] - FullTriangle[i, n_row - i - 1] for i in range(n_row)
        ])

        expression = lambda i: (FullTriangle[i, -1] ** 2) * np.sum(
            [
                (self.Deviations[k - 1] / self.DevFactors[k - 1]) ** 2 * (
                        1 / FullTriangle[i, k - 1] + 1 / np.sum(FullTriangle[:n_row - k, k - 1]))
                for k in list(range(n_row - i, n_row))]
        )

        mses = np.array([
            np.sqrt(expression(i)) for i in range(n_row)
        ])

        return pd.DataFrame({
            "Year": self.FullTriangle.years,
            "Provision": prov,
            "MSE": mses
        })

    def __str__(self):
        """
        Return a string representation of the Chain Mack Model object.

        Returns:
            str: String representation of the object.
        """
        return f"This is a Chain Mack Model, with development factors: {self.DevFactors}\nAnd Deviations: {self.Deviations}\nAnd Full triangle estimated:\n{self.FullTriangle}"

class ChainMackGeneral:
    """
    ChainMackGeneral class for performing generalized Mack Chain Ladder analysis on triangle data.

    Attributes:
        DevFactors (numpy.ndarray): Development factors calculated during the fitting process.
        Deviations (numpy.ndarray): Deviation factors calculated during the fitting process.
        FullTriangle (Triangle): Triangle object containing the full development data after fitting.
        alpha (float): Parameter for adjusting the weighting in the estimation process.
    """

    def __init__(self, alpha=1):
        """
        Initializes an instance of the ChainMackGeneral class.

        Args:
            alpha (float, optional): Parameter for adjusting the weighting. Defaults to 1.
        """
        self.DevFactors = None
        self.Deviations = None
        self.FullTriangle = None
        self.alpha = alpha

    def fit(self, triangle: Triangle, estimation_method='mean'):
        """
        Fits the ChainMackGeneral model to the provided triangle data.
        For mathematical backgroud, see this [https://www.researchgate.net/publication/228480205_Stochastic_Claims_Reserving_in_General_Insurance/citations]

        Args:
            triangle (Triangle): Triangle object containing the development data.
            estimation_method (str, optional): Method for estimating development factors. 
                Options: 'mean', 'median', or 'Qxx' (where 'xx' is the percentile to exclude). Defaults to 'mean'.

        Returns:
            None
        """

        n_row, n_col = triangle.Cum.shape
        DevFact = np.zeros(n_col - 1)
        Sigmas = np.zeros(n_col - 1)
        error = False

        for j in range(n_col - 1):
            if estimation_method == 'mean':
                DevFact[j] = np.average(triangle.Cum[:n_row - j - 1, j + 1] / triangle.Cum[:n_row - j - 1, j],
                                        weights=triangle.Cum[:n_row - j - 1, j] ** self.alpha)
            elif estimation_method == 'median':
                lst = triangle.Cum[:n_row - j - 1, j + 1] / triangle.Cum[:n_row - j - 1, j] * \
                      triangle.Cum[:n_row - j - 1, j] ** self.alpha / \
                      np.sum(triangle.Cum[:n_row - j - 1, j] ** self.alpha)
                i_factor = sorted(range(len(lst)), key=lambda i: lst[i])[len(lst) // 2]
                DevFact[j] = (triangle.Cum[:n_row - j - 1, j + 1] / triangle.Cum[:n_row - j - 1, j])[i_factor]
            elif estimation_method[0] == 'Q':
                alpha = float(estimation_method[1:]) / 100
                x = triangle.Cum[:n_row - j - 1, j + 1] / triangle.Cum[:n_row - j - 1, j]
                y = triangle.Cum[:n_row - j - 1, j] ** self.alpha
                sorted_x = sorted(x)[:int(len(x) * (1 - alpha))]
                x_pop = []
                y_pop = []
                for i in range(len(x)):
                    if x[i] in sorted_x:
                        x_pop.append(x[i])
                        y_pop.append(y[i])
                if len(x_pop) > 0:
                    DevFact[j] = np.average(x_pop, weights=y_pop)
                else:
                    DevFact[j] = np.average(x, weights=y)
            else:
                error = True

            if j < n_col - 1 - 1:
                Sigmas[j] = np.sqrt(1 / (n_row - j - 1 - 1) * np.sum(
                    triangle.Cum[:(n_row - j - 1), j] ** self.alpha *
                    np.sum(triangle.Cum[:n_row - j - 1, j + 1] / triangle.Cum[:n_row - j - 1, j] - DevFact[j]) ** 2))

        Sigmas[n_col - 1 - 1] = min(min(Sigmas[n_col - 1 - 2], Sigmas[n_col - 1 - 3]),
                                     Sigmas[n_col - 1 - 2] ** 2 / Sigmas[n_col - 1 - 3])

        if error:
            print("Estimation method should be one of these: 'mean' for mean of development factors, "
                  "'median' for the median, or like 'Q05' to eliminate highest 5% development factors, "
                  "which could be anomalous.")

        FullTriangle = triangle.Cum.copy()
        for i in range(n_row - 1, n_col - n_row - 1, -1):
            FullTriangle[i, n_row - i:] = FullTriangle[i, n_row - i - 1] * np.cumprod(DevFact[n_row - i - 1:])

        self.DevFactors = DevFact
        self.Deviations = Sigmas
        self.FullTriangle = Triangle(years=triangle.years, data=FullTriangle, isCumul=True)

    def Provisions(self):
        """
        Calculate the provisions for each year based on the fitted model.

        Returns:
            pandas.DataFrame: DataFrame containing the years, provisions, and mean squared errors.
        """

        FullTriangle = self.FullTriangle.Cum
        n_row, _ = FullTriangle.shape

        prov = np.array([FullTriangle[i, -1] - FullTriangle[i, n_row - i - 1] for i in range(n_row)])

        expression = lambda i: (FullTriangle[i, -1] ** 2) * np.sum(
            [(self.Deviations[k - 1] / self.DevFactors[k - 1]) ** 2 * (
                        1 / FullTriangle[i, k - 1] ** self.alpha + 1 / np.sum(
                    FullTriangle[:n_row - k, k - 1] ** self.alpha)) for k in list(range(n_row - i, n_row))])

        mses = np.array([np.sqrt(expression(i)) for i in range(n_row)])

        return pd.DataFrame({
            "Year": self.FullTriangle.years,
            "Provision": prov,
            "MSE": mses
        })


    def __str__(self):
        """
        Returns a string representation of the ChainMackGeneral model.

        Returns:
            str: String representation of the model.
        """
        return f"This is a Chain Ladder Model, with development factors: {self.DevFactors}\n" \
               f"And Deviations: {self.Deviations}\n" \
               f"And Full triangle estimated:\n{self.FullTriangle}"

class ChainGLM :
    """
    ChainGLM class for performing GLM version that replicate results of chain ladder analysis.

    Attributes:
        FullTriangle (Triangle): Triangle object containing the full development data after fitting.
    """
    def __init__(self,distribution = 'poisson',IncrementalModel = True):
        """
        Initializes an instance of the ChainGLM class.
        """
        self.Intercept    = None
        self.EffectDev    = None
        self.EffectYear   = None
        self.FullTriangle = None
        self.glmresult    = None
        self.dist         = distribution
        self.IncrModel    = IncrementalModel
        
    def fit(self, triangle: Triangle):
        """
        Fits the ChainGLM model to the provided triangle data.

        Args:
            triangle (Triangle): Triangle object containing the development data.

        Returns:
            None
        """

        if self.dist == "poisson":
            family = sm.families.Poisson (link=sm.families.links.Log())
        elif self.dist == "gamma":
            family = sm.families.Gamma   (link=sm.families.links.Log())
        elif self.dist == "normal":
            family = sm.families.Gaussian(link=sm.families.links.Log())


        n_row, n_col = triangle.Cum.shape
        df = pd.DataFrame({
            'year': np.repeat(range(triangle.Inc.shape[0]), triangle.Inc.shape[1]),
            'dev': np.tile(range(triangle.Inc.shape[1]), triangle.Inc.shape[0]),
            'value': triangle.Cum.flatten() if self.IncrModel else triangle.Inc.flatten()
        })

        df['year'] = pd.Categorical(df['year'])
        df['dev']  = pd.Categorical(df['dev'] )

        model = sm.GLM.from_formula('value ~ year + dev', data=df.dropna(), family=family, missing='drop')
        result = model.fit()

        df['predictions'] =  result.predict(df)
        df['dev'] = df['dev'].cat.codes
        df['year'] = df['year'].cat.codes
        df.loc[df['dev']+ df['year'] < n_row, 'predictions'] = df.loc[df['dev']+ df['year'] < n_row, 'value']

        self.FullTriangle = Triangle(triangle.years,data = df.pivot(index='year', columns='dev', values='predictions').values,isCumul=self.IncrModel)
        self.Intercept  = result.params[0]
        self.EffectDev  = result.params[n_row:]
        self.EffectYear = result.params[1:n_row] 
        self.glmresult  = result

    def refit(self, triangle:Triangle) : 
        if self.dist == "poisson":
            family = sm.families.Poisson (link=sm.families.links.Log())
        elif self.dist == "gamma":
            family = sm.families.Gamma   (link=sm.families.links.Log())
        elif self.dist == "normal":
            family = sm.families.Gaussian(link=sm.families.links.Log())

        n_row, n_col = triangle.Cum.shape
        df = pd.DataFrame({
            'year': np.repeat(range(triangle.Inc.shape[0]), triangle.Inc.shape[1]),
            'dev': np.tile(range(triangle.Inc.shape[1]), triangle.Inc.shape[0]),
            'value': triangle.Cum.flatten() if self.IncrModel else triangle.Inc.flatten()
        })

        df['year'] = pd.Categorical(df['year'])
        df['dev']  = pd.Categorical(df['dev']  )

        model = sm.GLM.from_formula('value ~ year + dev', data=df.dropna(), family=family, missing='drop')
        result = model.fit()
        
        df['predictions'] =  result.predict(df)
        df['dev'] = df['dev'].cat.codes
        df['year'] = df['year'].cat.codes
        df.loc[df['dev']+ df['year'] < n_row, 'predictions'] = df.loc[df['dev']+ df['year'] < n_row, 'value']

        return Triangle(triangle.years,data = df.pivot(index='year', columns='dev', values='predictions').values,isCumul=self.IncrModel)

    def BackEstimates(self) :
        df = pd.DataFrame({
            'year': np.repeat(range(self.FullTriangle.Inc.shape[0]), self.FullTriangle.Inc.shape[1]),
            'dev': np.tile(range(self.FullTriangle.Inc.shape[1]), self.FullTriangle.Inc.shape[0]),
        })
        df['predictions'] =  self.glmresult.predict(df)
        return Triangle(self.FullTriangle.years,data = df.pivot(index='year', columns='dev', values='predictions').values,isCumul=self.IncrModel)
    
    def Residuals(self) : 
        n_row, n_col = self.FullTriangle.Inc.shape
        df = pd.DataFrame({
            'year': np.repeat(range(self.FullTriangle.Inc.shape[0]), self.FullTriangle.Inc.shape[1]),
            'dev': np.tile(range(self.FullTriangle.Inc.shape[1]), self.FullTriangle.Inc.shape[0]),
        })
        k = n_col*(n_row+n_row-n_col + 1)/2
        p = n_row+n_col + 1
        df['residuals']   = self.glmresult.resid_pearson/np.sqrt(self.glmresult.scale)*np.sqrt(k/(k-p+2))
        return  df.pivot(index='year', columns='dev', values='residuals').values

    def Simulate(self,method = "Bootstrap" ) : 
        if method == "Bootstrap" : 
            resids = self.Residuals()
            mask = (resids != 0) & (~np.isnan(resids))
            shuffled_matrix = resids.copy()
            shuffled_matrix[mask] = np.random.permutation(resids[mask])
            estimates  = self.BackEstimates().Inc if self.IncrModel else self.BackEstimates().Cum
            simulation = estimates + np.sqrt(estimates)*shuffled_matrix 
            simulation = np.where(simulation < estimates, estimates, simulation)    
            return self.refit(Triangle(years=self.FullTriangle.years,data=simulation,isCumul=self.IncrModel))
        elif method == "Parametric Distribution" :
            n_row, n_col = self.FullTriangle.Cum.shape
            df = pd.DataFrame({
                'year': np.repeat(range(self.FullTriangle.Inc.shape[0]), self.FullTriangle.Inc.shape[1]),
                'dev': np.tile(range(self.FullTriangle.Inc.shape[1]), self.FullTriangle.Inc.shape[0]),
                'value': self.FullTriangle.Cum.flatten() if self.IncrModel else self.FullTriangle.Inc.flatten()
            })
            df['predictions'] =  simulate_GLM(self.glmresult, df[['year','dev']])
            df['predictions'] = df['predictions'].astype(float)
            df.loc[df['dev']+ df['year'] < n_row, 'predictions'] = df.loc[df['dev']+ df['year'] < n_row, 'value']
            return Triangle(self.FullTriangle.years,data = df.pivot(index='year', columns='dev', values='predictions').values,isCumul=self.IncrModel)

    def Provisions(self):
        """
        Calculate provisions of GLM Chain model for each year based on the full triangle data.

        Returns:
            pandas.DataFrame: DataFrame containing the year and corresponding provisions.
        """

        FullTriangle = self.FullTriangle.Cum
        n_row, n_col = FullTriangle.shape
        prov = np.array([
            FullTriangle[i, -1] - FullTriangle[i, n_row - i - 1] for i in range(n_row)
        ])
        return pd.DataFrame({
            "Year": self.FullTriangle.years,
            "Provision": prov
        })
    
    def __str__(self):
        """
        Return a string representation of the GLM Chain Model object.

        Returns:
            str: String representation of the object.
        """
        return f"This is a GLM Chain Model, with distribution {self.dist}\nwith intercept: {self.Intercept}\n,with years effect: \n{self.EffectYear}\n,with developement effect: \n{self.EffectYear}\nAnd Full triangle estimated:\n{self.FullTriangle}"


