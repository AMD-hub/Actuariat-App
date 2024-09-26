from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from random import shuffle
import statsmodels.api as sm


class Triangle:
    def __init__(self, years, data, isCumul=True):
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

        # Calculate development factors
        n_row, n_col = triangle.Cum.shape
        DevFact = np.zeros(n_col - 1)
        for j in range(n_col - 1):
            DevFact[j] = np.sum(triangle.Cum[:n_row - j - 1, j + 1]) / np.sum(triangle.Cum[:n_row - j - 1, j])

        # Project full triangle using development factors
        FullTriangle = triangle.Cum.copy()
        for j in range(n_col - 1):
            FullTriangle[n_row - j - 1:, j + 1] = FullTriangle[n_row - j - 1:, j] * DevFact[j]

        # Update class attributes
        self.DevFactors = DevFact
        self.FullTriangle = Triangle(years=triangle.years, data=FullTriangle, isCumul=True)

    def Plot_by_dev(self, dev, path=None):
        """
        Plot the relationship between cumulative values for a specific development period (dev) 
        and the next cumulative values at year i+1.

        Args:
            dev (int): Development period to plot.
            path (str, optional): Path to save the plot image. Defaults to None.

        Returns:
            tuple: Tuple containing two axes objects representing the scatter plots.
        """

        FullTriangle = self.FullTriangle.Cum
        years = np.array(self.FullTriangle.years)

        n_row, n_col = FullTriangle.shape
        j = dev

        x = FullTriangle[:n_row - j - 1, j]
        y = FullTriangle[:n_row - j - 1, j + 1]
        yrs = years[:n_row - j - 1]

        x_est = FullTriangle[n_row - j - 1:, j]
        y_est = FullTriangle[n_row - j - 1:, j + 1]
        yrs_est = years[n_row - j - 1:]
        l = np.arange(0, max(max(x), max(x_est)) * 1.05)

        fig, (ax, axx) = plt.subplots(1, 2)
        fig.suptitle(f'Testing Linearity on development dev = {j}', fontsize=20, fontname='serif')
        fig.set_size_inches(10, 5)

        ax.scatter(x=x, y=y, marker='*', c='blue', alpha=0.8, s=100, label="True value")
        ax.scatter(x=x_est, y=y_est, marker='*', c='red', alpha=0.8, s=100, label="Estimated value")
        ax.plot(l, self.DevFactors[j] * l, linestyle='-', color='purple')

        ax.set_xlabel('Cumul at year i')
        ax.set_ylabel('Next Cumul at year i')
        ax.legend(loc='upper left')
        ax.grid()

        axx.scatter(x=yrs, y=y / x, marker='*', c='blue', alpha=0.8, s=100, label="True value")
        axx.scatter(x=yrs_est, y=y_est / x_est, marker='*', c='red', alpha=0.8, s=100, label="Estimated value")
        axx.plot(years, self.DevFactors[j] * np.ones_like(years), linestyle='--', color='purple')

        axx.set_xlabel('Cumul at year i')
        axx.set_ylabel('Next Cumul at year i')
        axx.legend(loc='upper right')
        axx.grid()

        if path is not None:
            fig.savefig(path)

        return (ax, axx)

    def Plot_all(self, path=None):
        """
        Plot the relationship between cumulative values for all development periods 
        and the next cumulative values at year i+1 for each development period.

        Args:
            path (str, optional): Path to save the plot image. Defaults to None.

        Returns:
            None
        """

        FullTriangle = self.FullTriangle.Cum
        years = np.array(self.FullTriangle.years)
        n_row, n_col = FullTriangle.shape

        fig, ax = plt.subplots(n_row - 1, 2, gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
        fig.set_size_inches(10, 5 * (n_row - 1))

        for j in range(n_row - 1):
            x = FullTriangle[:n_row - j - 1, j]
            y = FullTriangle[:n_row - j - 1, j + 1]
            yrs = years[:n_row - j - 1]
            x_est = FullTriangle[n_row - j - 1:, j]
            y_est = FullTriangle[n_row - j - 1:, j + 1]
            yrs_est = years[n_row - j - 1:]
            l = np.arange(0, max(max(x), max(x_est)) * 1.05)

            ax[j, 0].scatter(x=x, y=y, marker='*', c='blue', alpha=0.8, s=100, label="True value")
            ax[j, 0].scatter(x=x_est, y=y_est, marker='*', c='red', alpha=0.8, s=100, label="Estimated value")
            ax[j, 0].plot(l, self.DevFactors[j] * l, linestyle='-', color='purple')
            ax[j, 0].set_xlabel('Cumul at year i')
            ax[j, 0].set_ylabel('Next Cumul at year i')
            ax[j, 0].legend(loc='upper left')
            ax[j, 0].grid()
            ax[j, 0].set_title(f'Plotting cumuls for dev {j}')

            ax[j, 1].scatter(x=yrs, y=y / x, marker='*', c='blue', alpha=0.8, s=100, label="True value")
            ax[j, 1].scatter(x=yrs_est, y=y_est / x_est, marker='*', c='red', alpha=0.8, s=100, label="Estimated value")
            ax[j, 1].plot(years, self.DevFactors[j] * np.ones_like(years), linestyle='--', color='purple')
            ax[j, 1].set_xlabel('Cumul at year i')
            ax[j, 1].set_ylabel('Next Cumul at year i')
            ax[j, 1].legend(loc='upper right')
            ax[j, 1].grid()
            ax[j, 1].set_title(f'Plotting dev factors for dev {j}')

        if path is not None:
            fig.savefig(path)

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

    def Plot_Provisions(self, path=None):
        """
        Plot the evolution of provisions over the years.

        Args:
            path (str, optional): Path to save the plot image. Defaults to None.

        Returns:
            None
        """

        data = self.Provisions()

        fig, ax = plt.subplots()
        fig.suptitle(f'Evolution of Provisions by years', fontsize=20, fontname='serif')
        fig.set_size_inches(10, 10)

        ax.scatter(x=data['Year'], y=data['Provision'], marker='*', c='blue', alpha=0.8, s=100, label="Provisions")
        ax.plot(data['Year'], data['Provision'], linestyle='--', c='blue', alpha=0.5)

        ax.set_xlabel('Years')
        ax.set_ylabel('Provisions')
        ax.legend(loc='upper left')
        ax.grid()

        if path is not None:
            fig.savefig(path)

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
        Sigmas = np.zeros(n_col - 1)

        for j in range(n_col - 1):
            DevFact[j] = np.sum(triangle.Cum[:n_row - j - 1, j + 1]) / np.sum(triangle.Cum[:n_row - j - 1, j])
            if j < n_col - 1 - 1:
                Sigmas[j] = np.sqrt(1 / (n_row - j - 1 - 1) * np.sum(
                    triangle.Cum[:(n_row - j - 1), j] * (triangle.Cum[:n_row - j - 1, j + 1]) / np.sum(
                        triangle.Cum[:n_row - j - 1, j] - DevFact[j]) ** 2))
        Sigmas[n_col - 1 - 1] = min(min(Sigmas[n_col - 1 - 2], Sigmas[n_col - 1 - 3]),
                                     Sigmas[n_col - 1 - 2] ** 2 / Sigmas[n_col - 1 - 3])

        FullTriangle = triangle.Cum.copy()
        for i in range(n_row - 1, n_col - n_row - 1, -1):
            FullTriangle[i, n_row - i:] = FullTriangle[i, n_row - i - 1] * np.cumprod(DevFact[n_row - i - 1:])

        self.DevFactors = DevFact
        self.Deviations = Sigmas
        self.FullTriangle = Triangle(years=triangle.years, data=FullTriangle, isCumul=True)

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

    def Plot_by_dev(self, dev, path=None):
        """
        Plot the relationship between cumulative values and the next cumulative values
        for a specific development period using the ChainMackGeneral method.

        Args:
            dev (int): The development period to plot.
            path (str, optional): Path to save the plot image. Defaults to None.

        Returns:
            None
        """

        FullTriangle = self.FullTriangle.Cum
        years = np.array(self.FullTriangle.years)

        n_row, n_col = FullTriangle.shape
        j = dev

        x = FullTriangle[:n_row - j - 1, j]
        y = FullTriangle[:n_row - j - 1, j + 1]
        yrs = years[:n_row - j - 1]

        x_est = FullTriangle[n_row - j - 1:, j]
        y_est = FullTriangle[n_row - j - 1:, j + 1]
        yrs_est = years[n_row - j - 1:]
        l = np.arange(0, max(max(x), max(x_est)) * 1.05)

        fig, (ax, axx) = plt.subplots(1, 2)
        fig.suptitle(f'Testing Linearity on development dev = {j}', fontsize=20, fontname='serif')
        fig.set_size_inches(10, 5)

        ax.scatter(x=x, y=y, marker='*', c='blue', alpha=0.8, s=100, label="True value")
        ax.scatter(x=x_est, y=y_est, marker='*', c='red', alpha=0.8, s=100, label="Estimated value")
        ax.plot(l, self.DevFactors[j] * l, linestyle='-', color='purple')
        ax.fill_between(l, self.DevFactors[j] * l + 1.95 * self.Deviations[j] * np.sqrt(l),
                        self.DevFactors[j] * l - 1.95 * self.Deviations[j] * np.sqrt(l),
                        color='gray', alpha=0.3, label='Confidence interval')

        ax.set_xlabel('Cumul at year i')
        ax.set_ylabel('Next Cumul at year i')
        ax.legend(loc='upper left')
        ax.grid()

        axx.scatter(x=yrs, y=y / x, marker='*', c='blue', alpha=0.8, s=100, label="True value")
        axx.scatter(x=yrs_est, y=y_est / x_est, marker='*', c='red', alpha=0.8, s=100, label="Estimated value")
        axx.plot(years, self.DevFactors[j] * np.ones_like(years), linestyle='--', color='purple')

        axx.set_xlabel('Cumul at year i')
        axx.set_ylabel('Next Cumul at year i')
        axx.legend(loc='upper right')
        axx.grid()

        if path is not None:
            fig.savefig(path)

    def Plot_all(self, path=None):
        """
        Plot the relationship between cumulative values and the next cumulative values
        for all development periods using the ChainMackGeneral method.

        Args:
            path (str, optional): Path to save the plot image. Defaults to None.

        Returns:
            None
        """

        FullTriangle = self.FullTriangle.Cum
        years = np.array(self.FullTriangle.years)
        n_row, n_col = FullTriangle.shape

        fig, ax = plt.subplots(n_row - 1, 2, gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
        fig.set_size_inches(10, 5 * (n_row - 1))

        for j in range(n_row - 1):
            x = FullTriangle[:n_row - j - 1, j]
            y = FullTriangle[:n_row - j - 1, j + 1]
            yrs = years[:n_row - j - 1]
            x_est = FullTriangle[n_row - j - 1:, j]
            y_est = FullTriangle[n_row - j - 1:, j + 1]
            yrs_est = years[n_row - j - 1:]
            l = np.arange(0, max(max(x), max(x_est)) * 1.05)

            ax[j, 0].scatter(x=x, y=y, marker='*', c='blue', alpha=0.8, s=100, label="True value")
            ax[j, 0].scatter(x=x_est, y=y_est, marker='*', c='red', alpha=0.8, s=100, label="Estimated value")
            ax[j, 0].plot(l, self.DevFactors[j] * l, linestyle='-', color='purple')
            ax[j, 0].fill_between(l, self.DevFactors[j] * l + 1.95 * self.Deviations[j] * np.sqrt(l),
                                   self.DevFactors[j] * l - 1.95 * self.Deviations[j] * np.sqrt(l),
                                   color='gray', alpha=0.3, label='Confidence interval')
            ax[j, 0].set_xlabel('Cumul at year i')
            ax[j, 0].set_ylabel('Next Cumul at year i')
            ax[j, 0].legend(loc='upper left')
            ax[j, 0].grid()
            ax[j, 0].set_title(f'Plotting cumuls for dev {j}')

            ax[j, 1].scatter(x=yrs, y=y / x, marker='*', c='blue', alpha=0.8, s=100, label="True value")
            ax[j, 1].scatter(x=yrs_est, y=y_est / x_est, marker='*', c='red', alpha=0.8, s=100, label="Estimated value")
            ax[j, 1].plot(years, self.DevFactors[j] * np.ones_like(years), linestyle='--', color='purple')
            ax[j, 1].set_xlabel('Cumul at year i')
            ax[j, 1].set_ylabel('Next Cumul at year i')
            ax[j, 1].legend(loc='upper right')
            ax[j, 1].grid()
            ax[j, 1].set_title(f'Plotting dev factors for dev {j}')

        if path is not None:
            fig.savefig(path)

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

    def Plot_Provisions(self, path=None):
        """
        Plot the evolution of provisions over the years.

        Args:
            path (str, optional): Path to save the plot image. Defaults to None.

        Returns:
            None
        """

        data = self.Provisions()

        fig, ax = plt.subplots()
        fig.suptitle(f'Evolution of Provisions by years', fontsize=20, fontname='serif')
        fig.set_size_inches(10, 10)

        ax.scatter(x=data['Year'], y=data['Provision'], marker='*', c='blue', alpha=0.8, s=100, label="Provisions")
        ax.plot(data['Year'], data['Provision'], linestyle='--', c='blue', alpha=0.5)

        ax.fill_between(data['Year'], data['Provision'] - 1.95 * data['MSE'], data['Provision'] + 1.95 * data['MSE'],
                        color='gray', alpha=0.3, label='Confidence interval')

        ax.set_xlabel('Years')
        ax.set_ylabel('Provisions')
        ax.legend(loc='upper left')
        ax.grid()

        if path is not None:
            fig.savefig(path)

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
    def __init__(self,distribution = 'poisson'):
        """
        Initializes an instance of the ChainGLM class.
        """
        self.Intercept    = None
        self.EffectDev    = None
        self.EffectYear   = None
        self.FullTriangle = None
        self.Residuals    = None
        self.VarPred      = None
        self.glmresult    = None
        self.dist         = distribution
        
    def fit(self, triangle: Triangle):
        """
        Fits the ChainGLM model to the provided triangle data.

        Args:
            triangle (Triangle): Triangle object containing the development data.

        Returns:
            None
        """

        if self.dist == "poisson":
            family = sm.families.Poisson(link=sm.families.links.Log())
        elif self.dist == "gamma":
            family = sm.families.Gamma(link=sm.families.links.Log())

        # Calculate development factors
        n_row, n_col = triangle.Cum.shape

        df = pd.DataFrame({
            'year': np.repeat(range(triangle.Inc.shape[0]), triangle.Inc.shape[1]),
            'dev': np.tile(range(triangle.Inc.shape[1]), triangle.Inc.shape[0]),
            'incr': triangle.Inc.flatten()
        })

        df['year'] = pd.Categorical(df['year'])
        df['dev'] = pd.Categorical(df['dev']  )

        model = sm.GLM.from_formula('incr ~ year + dev', data=df.dropna(), family=family, missing='drop')
        result = model.fit()

        k = n_col*(n_row+n_row-n_col + 1)/2
        p = n_row+n_col + 1

        df['predictions'] =  result.predict(df)
        df['residuals']   = result.resid_pearson/np.sqrt(result.scale)*np.sqrt(k/(k-p+2))
        df['variance']    =  np.nan
        df.loc[~np.isnan(df['incr']), 'variance']= result.get_prediction().var_pred
        self.FullTriangle = Triangle(triangle.years,data = df.pivot(index='year', columns='dev', values='predictions').values,isCumul=False)
        self.Residuals  = df.pivot(index='year', columns='dev', values='residuals').values
        self.VarPred    = df.pivot(index='year', columns='dev', values='variance').values
        self.Intercept  = result.params[0]
        self.EffectDev  = result.params[n_row:]
        self.EffectYear = result.params[1:n_row] 

        self.glmresult = result


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

class ChainLadderBoot : 
    def __init__(self,Simulation_Number) :
        self.DevFactors     = None 
        self.FullTriangle   = None 
        self.SimProvisions  = None
        self.SimProvisionsTotal = None
        self.glmchain       = None
        self.NumSim         = Simulation_Number

    def fit(self,triangle:Triangle,num_simulations = None) : 
        if num_simulations == None :
            num_simulations = self.NumSim 
        glm = ChainGLM() 
        glm.fit(triangle)
        self.glmchain = glm
        
        Residus = glm.Residuals
        Predus  = glm.FullTriangle.Inc
        Varus   = glm.VarPred 

        Factors = []
        SimProvisions = [] 
        Haha = np.array(Residus[:-1,:-1].reshape(1,(Residus.shape[0]-1)*(Residus.shape[1]-1)))[0].copy()
        for _ in range(num_simulations) :
            NonNullElements = [y for y in Haha if not np.isnan(y) ]
            shuffle(NonNullElements)
            Y = Residus.copy()
            cpt = 0 
            for i in range(Y[:-1,:-1].shape[0]) : 
                for j in range(Y[:-1,:-1].shape[1]) : 
                    if not np.isnan(Y[i,j]) :
                        Y[i,j] = NonNullElements[cpt]
                        cpt+=1
                        
            Z_b = Predus + np.multiply(np.sqrt(Varus),Y)

            chainSim = ChainLadder()
            chainSim.fit(Triangle(years=triangle.years,data=Z_b,isCumul=False))   
            Factors.append(chainSim.DevFactors)
            SimProvisions.append(chainSim.Provisions()['Provision'].values)

        Factors = np.array(Factors)
        self.DevFactors  = np.mean(Factors,axis=0)
        self.SimProvisions = np.array(SimProvisions)
        self.SimProvisionsTotal = np.sum(self.SimProvisions,axis=1)

        
        n_row,n_col = triangle.shape
        FullTriangle = triangle.Cum.copy()
        for j in range(n_col-1) : 
            FullTriangle[n_row-j-1:,j+1] = FullTriangle[n_row-j-1:,j]*self.DevFactors[j] 
        
        self.FullTriangle = Triangle(years=triangle.years,data=FullTriangle,isCumul=True) 



    def Plot_by_dev(self,dev,path=None) : 
        FullTriangle = self.FullTriangle.Cum
        years =np.array(self.FullTriangle.years)

        n_row,n_col = FullTriangle.shape
        j = dev

        x =  FullTriangle[:n_row-j-1,j]  
        y =  FullTriangle[:n_row-j-1,j+1]
        yrs = years[:n_row-j-1]

        x_est = FullTriangle[n_row-j-1:,j]  
        y_est = FullTriangle[n_row-j-1:,j+1]
        yrs_est = years[n_row-j-1:]
        l = np.arange(0,max(max(x),max(x_est))*1.05)


        fig, (ax,axx) = plt.subplots(1,2)
        fig.suptitle(f'Testing Linearity on developpement dev = {j}', fontsize=20, fontname='serif')
        fig.set_size_inches(10,5)

        ax.scatter(x=x, y=y, marker='*', c='blue', alpha=0.8,s=100,label = "True value")
        ax.scatter(x=x_est, y=y_est, marker='*', c='red', alpha=0.8,s=100,label = "Estimated value")
        ax.plot(l, self.DevFactors[j]*l ,linestyle='-',color = 'purple')

        ax.set_xlabel('Cumul at year i')
        ax.set_ylabel('Next Cumul at year i')
        ax.legend(loc='upper left')
        ax.grid()

        axx.scatter(x=yrs, y=y/x, marker='*', c='blue', alpha=0.8,s=100,label = "True value")
        axx.scatter(x=yrs_est, y=y_est/x_est, marker='*', c='red', alpha=0.8,s=100,label = "Estimated value")
        axx.plot(years, self.DevFactors[j]*np.ones_like(years) ,linestyle='--',color = 'purple')

        axx.set_xlabel('Cumul at year i')
        axx.set_ylabel('Next Cumul at year i')
        axx.legend(loc='upper right')
        axx.grid()

        if path != None :
            fig.savefig(path)
        
        return (ax,axx)

    def Plot_all(self,path=None) :
        FullTriangle = self.FullTriangle.Cum
        years =np.array(self.FullTriangle.years)
        n_row,n_col = FullTriangle.shape

        fig, ax = plt.subplots(n_row-1,2,gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
        fig.set_size_inches(10,5*(n_row-1))
        # fig.suptitle(f'Testing Linearity for all developpements', fontsize=10, fontname='serif')

        for j in range(n_row-1) :
            x =  FullTriangle[:n_row-j-1,j]  
            y =  FullTriangle[:n_row-j-1,j+1]
            yrs = years[:n_row-j-1]
            x_est = FullTriangle[n_row-j-1:,j]  
            y_est = FullTriangle[n_row-j-1:,j+1]
            yrs_est = years[n_row-j-1:]
            l = np.arange(0,max(max(x),max(x_est))*1.05)

            ax[j,0].scatter(x=x, y=y, marker='*', c='blue', alpha=0.8,s=100,label = "True value")
            ax[j,0].scatter(x=x_est, y=y_est, marker='*', c='red', alpha=0.8,s=100,label = "Estimated value")
            ax[j,0].plot(l, self.DevFactors[j]*l ,linestyle='-',color = 'purple')
            ax[j,0].set_xlabel('Cumul at year i')
            ax[j,0].set_ylabel('Next Cumul at year i')
            ax[j,0].legend(loc='upper left')
            ax[j,0].grid()
            ax[j,0].set_title(f'Plotting cumuls for dev {j}')


            ax[j,1].scatter(x=yrs, y=y/x, marker='*', c='blue', alpha=0.8,s=100,label = "True value")
            ax[j,1].scatter(x=yrs_est, y=y_est/x_est, marker='*', c='red', alpha=0.8,s=100,label = "Estimated value")
            ax[j,1].plot(years, self.DevFactors[j]*np.ones_like(years) ,linestyle='--',color = 'purple')
            ax[j,1].set_xlabel('Cumul at year i')
            ax[j,1].set_ylabel('Next Cumul at year i')
            ax[j,1].legend(loc='upper right')
            ax[j,1].grid()
            ax[j,1].set_title(f'Plotting dev factrs for dev {j}')

        if path != None :
            fig.savefig(path)

         
    def Provisions(self) : 
        FullTriangle = self.FullTriangle.Cum
        n_row,n_col = FullTriangle.shape
        prov = np.array([ 
            FullTriangle[i,-1] - FullTriangle[i,n_row-i-1] for i in range(n_row)
        ])
        return pd.DataFrame({
            "Year" : self.FullTriangle.years,
            "Provision" : prov
        })

    def Plot_Provisions(self,path=None) : 
        data = self.Provisions()

        fig, ax = plt.subplots()
        fig.suptitle(f'Evolution of Provisions by years', fontsize=20, fontname='serif')
        fig.set_size_inches(10,10)

        ax.scatter(x=data['Year'], y=data['Provision'], marker='*', c='blue', alpha=0.8,s=100,label = "Provisions")
        ax.plot(data['Year'], data['Provision'], linestyle='--', c='blue', alpha=0.5)

        ax.set_xlabel('Years')
        ax.set_ylabel('Provisions')
        ax.legend(loc='upper left')
        ax.grid()


        if path != None :
            fig.savefig(path)
    

    def __str__(self) :
        return f"This is a BootStrap Chain Ladder Model, with {self.NumSim} with developpement factors : {self.DevFactors}\nAnd Full triangle estimated :\n{self.FullTriangle}"

