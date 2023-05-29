import bilby
import numpy as np

class MorseFactorPrior(bilby.prior.Prior):
    """
    Class definig the discrete prior that should be used
    for the Morse factor when analyzing a single event.
    The Morse factor should be 0, 0.5 or 1, for type I, II,
    and III images, respectively
    """

    def __init__(self, name = None, latex_label = None):
        """
        Initialization function.
        We do not precise controlable minimum or maximum
        values as the Morse factor can only tale fixed values.
        This is written such as to have a uniform probability
        for integers 0, 1 and 2, but returns half of their value

        ARGS:
        -----
        - name: the name of the parameters (typically 'n_phase')
        - latex_label: latex label used for the event (ex: '$n_{1}$')
        """
        bilby.prior.Prior.__init__(self, name, latex_label)
        self.minimum = 0
        self.maximum = 2 

    def rescale(self, val):
        """
        Maps the continuous 0 to 1 interval to the 0, 0.5, 1
        discrete distribution
        """
        resc = np.floor((self.maximum+1)*val)/2.
        return resc

    def prob(self, val):
        """
        Compute the probability of having a given value in the interval.
        This is 1/3 for each value here
        """
        return ((val >= 0) & (val <= self.maximum/2.))/float(self.maximum+1.) * (np.modf(2*val)[0] == 0).astype(int)

    def cdf(self, val):
        """
        Representation of the cumulative density function which can 
        be used in order to make the samples. 
        """
        cdf = (val <= self.maximum/2.) * (np.floor(2*val)+1)/float(self.maximum+1) + (val > self.maximum/2.)
        return cdf 

class MorseFactorDifference(bilby.prior.Prior):
    """
    Class definig the prior for the Morse factor difference.
    This is done by using the symmetry, with a difference of -1
    being equal to 1 and a difference of -0.5 bein equal to 1.5.
    So, we have a discrete uniform between 0, 0.5, 1, and 1.5

    ARGS:
    -----
    - name: the name of the parameter (typically 'delta_n')
    - latex_label: label that should be used in the plot

    """
    def __init__(self, name = None, latex_label = None):
        """
        Initialization function. The maximum and minimum
        are fixed as we know that the Morse factor difference
        can only be a given set of value
        """
        bilby.prior.Prior.__init__(self, name, latex_label)
        self.minimum = 0
        self.maximum = 3 

    def rescale(self, val):
        """
        Function mapping  the interval 0 to 1 to the 
        discrete set of values 0, 0.5, 1, 1.5
        """
        resc = np.floor((self.maximum+1)*val)/2.
        return resc

    def prob(self, val):
        """
        Function comuting the probability to get 
        each of the values in the set. It is equivalent to 
        1/4 for all values
        """
        return ((val >= 0) & (val <= self.maximum/2.))/float(self.maximum + 1) * (np.modf(2*val)[0] == 0).astype(int)

    def cdf(self, val):
        """
        Function computing the cumulative distribution 
        function for the uniform distribution considered 
        in the case of the difference in Morse factor
        """
        cdf = (val <= self.maximum/2.) * (np.floor(2*val) + 1)/float(self.maximum+1) + (val > self.maximum/2.)
        return cdf 
