import numpy as np
import librosa
import scipy.optimize as op
import matplotlib.pyplot as plt

def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.)

def hz2star(frequencies):
    
    frequencies = np.asanyarray(frequencies)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 /10 #used to be 10

    stars = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1600.0  # beginning of log region (Hz)
    min_log_star = (min_log_hz - f_min) / f_sp  # same (stars)
    logstep = np.log(6.4) / 150  # step size for log region, used to be 150

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        stars[log_t] = min_log_star + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        stars = min_log_star + np.log(frequencies / min_log_hz) / logstep

    return stars

def star2hz(stars):
    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 10 
    freqs = f_min + f_sp * stars

    # And now the nonlinear scale
    min_log_hz = 1600.0  # beginning of log region (Hz)
    min_log_star = (min_log_hz - f_min) / f_sp  # same (stars)
    logstep = np.log(6.4) / 150  # step size for log region

    try: 
        if stars.ndim:
            # If we have vector data, vectorize
            log_t = stars >= min_log_star
            freqs[log_t] = min_log_hz * np.exp(logstep * (stars[log_t] - min_log_star))
    except:
        if stars >= min_log_star:
        # If we have scalar data, check directly
            freqs = min_log_hz * np.exp(logstep * (stars - min_log_star))

    return freqs


def star_filterbank(
    sr, 
    n_fft, 
    n_stars=128,
    fmin=0.0,
    fmax=None,
    dtype=np.float32):
    
    
    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_stars = int(n_stars)
    weights = np.zeros((n_stars, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of star bands - uniformly spaced between limits
    star_f = star_frequencies(n_stars + 2, fmin=fmin, fmax=fmax)

    fdiff = np.diff(star_f)
    ramps = np.subtract.outer(star_f, fftfreqs)

    for i in range(n_stars):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    enorm = 2.0 / (star_f[2 : n_stars + 2] - star_f[:n_stars])
    weights *= enorm[:, np.newaxis]

    # Only check weights if f_star[0] is positive
    if not np.all((star_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        print(
            "Empty filters detected in star frequency basis. "
            "Some channels will produce empty responses. "
            "Try increasing your sampling rate (and fmax) or "
            "reducing n_stars."
        )

    return weights

def star_frequencies(n_stars=128, fmin=0.0, fmax=11025.0):
    # 'Center freqs' of star bands - uniformly spaced between limits
    min_star = hz2star(fmin)
    max_star = hz2star(fmax)

    stars = np.linspace(min_star, max_star, n_stars)

    return star2hz(stars)

def star_spectrogram(y=None, sr=44100, S=None, n_fft=2048, hop_length=512, 
                    win_length=None, 
                    window='hann',
                    center=True,
                    pad_mode="reflect",
                    power=2.0,
                    **kwargs,
                    ):
    S, n_fft = librosa.core.spectrum._spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Build a Star filter
    star_filters = star_filterbank(sr, n_fft, **kwargs)

    return np.dot(star_filters, S)

def rmse(A, B=None):
    if B is None:
        return np.sqrt(np.mean(np.square(A)))
    else:
        return np.sqrt(np.mean(np.square(A-B)))
    

    
# below are the 4pl and 5pl parameter approximation functions    
def twopl(x, y, color=None, **kwargs):
    data = kwargs.pop("data")

    result = fit_2pl(data[x].values, data[y].values.astype(np.double))
#     print(result)
    try:
        result_2pl = two_param_logistic(result)
        t = np.arange(128) + 1

        if color is None:
            lines, = plt.plot(x.mean(), y.mean())
            color = lines.get_color()
            lines.remove()

        plt.plot(t, result_2pl(t), color=color)
    except TypeError:
        pass

def fourpl(x, y, color=None, **kwargs):
    data = kwargs.pop("data")

    result = fit_4pl(data[x].values, data[y].values.astype(np.double))
#     print(result)
    try:
        result_4pl = four_param_logistic(result)
        t = np.arange(128) + 1

        if color is None:
            lines, = plt.plot(x.mean(), y.mean())
            color = lines.get_color()
            lines.remove()

        plt.plot(t, result_4pl(t), color=color)
    except TypeError:
        pass
    
def fivepl(x, y, color=None, **kwargs):
    data = kwargs.pop("data")

    result = fit_5pl(data[x].values, data[y].values.astype(np.double))
    try:
        result_5pl = five_param_logistic(result)
        t = np.arange(128) + 1

        if color is None:
            lines, = plt.plot(x.mean(), y.mean())
            color = lines.get_color()
            lines.remove()

        plt.plot(t, result_5pl(t), color=color)
    except TypeError:
        pass

    
def fit_2pl(x, y, p_start=None, verbose=False):
    """Fits a 2 parameter logistic function to the data.

    Args:
        x: a numpy array of length n
            assumes a finite number of unique x values
        y: a numpy array of length n
            must be of dtype double or float so multiplication works
    optional:
        p_start: an iterable of length 2 that would be a reasonable spot to
            start the optimization. If None, tries to estimate it.
            B, M = p_start
            default=None
        verbose: boolean flag that allows printing of more error messages.

    Returns:
        p_result: an iterable of length 4 that defines the model that
        is maximally likely
            A, K, B, M = p_result
    """
    try:
        if not p_start:
            p_start = est_pstart_2(x, y)
    except TypeError:
        pass
    for i in range(10):
        if verbose and i > 0:
            print("retry", i)
        result = op.minimize(
            nll_2,
            p_start,
            args=(x, y),
            jac=ndll_2,
            bounds=(
                (None, None),
                (None, None),
            ),
        )
        if result.success:
            return result.x
        else:
            if verbose:
                print(p_start, "failure", result)
            p_start = result.x
    return False    


def fit_4pl(x, y, p_start=None, verbose=False, epsilon=1e-16):
    """Fits a 4 parameter logistic function to the data.

    Args:
        x: a numpy array of length n
            assumes a finite number of unique x values
        y: a numpy array of length n
            must be of dtype double or float so multiplication works
    optional:
        p_start: an iterable of length 4 that would be a reasonable spot to
            start the optimization. If None, tries to estimate it.
            A, K, B, M = p_start
            default=None
        verbose: boolean flag that allows printing of more error messages.
        epsilon: limits A and K between (epsilon, 1 - epsilon) for stability

    Returns:
        p_result: an iterable of length 4 that defines the model that
        is maximally likely
            A, K, B, M = p_result
    """
    try:
        if not p_start:
            p_start = est_pstart(x, y)
    except TypeError:
        pass
    for i in range(10):
        if verbose and i > 0:
            print("retry", i)
        result = op.minimize(
            nll,
            p_start,
            args=(x, y),
            jac=ndll,
            bounds=(
                (epsilon, 1 - epsilon),
                (epsilon, 1 - epsilon),
                (None, None),
                (None, None),
            ),
        )
        if result.success:
            return result.x
        else:
            if verbose:
                print(p_start, "failure", result)
            p_start = result.x
    return False

def fit_5pl(x, y, p_start=None, verbose=False, epsilon=1e-16):
    """Fits a 5 parameter logistic function to the data.

    Args:
        x: a numpy array of length n
            assumes a finite number of unique x values
        y: a numpy array of length n
            must be of dtype double or float so multiplication works
    optional:
        p_start: an iterable of length 4 that would be a reasonable spot to
            start the optimization. If None, tries to estimate it.
            A, K, B, M, E = p_start
            default=None
        verbose: boolean flag that allows printing of more error messages.
        epsilon: limits A and K between (epsilon, 1 - epsilon) for stability

    Returns:
        p_result: an iterable of length 5 that defines the model that
        is maximally likely
            A, K, B, M, E = p_result
    """
    try:
        if not p_start:
            p_start = est_pstart_5(x, y)
    except TypeError:
        pass
    for i in range(3):
        if verbose and i > 0:
            print("retry", i)
        result = op.minimize(
            nll_5,
            p_start,
            args=(x, y),
            jac=ndll_5,
            bounds=(
                (epsilon, 1 - epsilon),
                (epsilon, 1 - epsilon),
                (None, None),
                (None, None),
                (None, None),
            ),
        )
        if result.success:
            return result.x
        else:
            if verbose:
                print(p_start, "failure", result)
            p_start = result.x
    return False
def two_param_logistic(p):
    """2p logistic function maker.

    Returns a function that accepts x and returns y for
    the 2-parameter logistic defined by p.

    The 2p logistic is defined by:
    y = 1 / ((1 + exp(-B*(x-M))))

    Args:
        p: an iterable of length 5
            A, K, B, M, E = p

    Returns:
        A function that accepts a numpy array as an argument
        for x values and returns the y values for the defined 5pl curve.
    """
    B, M = p

    def f(x):
        return 1 / (1 + np.exp(-B * (x - M)))

    return f

def four_param_logistic(p):
    """4p logistic function maker.

    Returns a function that accepts x and returns y for
    the 4-parameter logistic defined by p.

    The 4p logistic is defined by:
    y = A + (K - A) / (1 + exp(-B*(x-M)))

    Args:
        p: an iterable of length 4
            A, K, B, M = p

    Returns:
        A function that accepts a numpy array as an argument
        for x values and returns the y values for the defined 4pl curve.
    """
    A, K, B, M = p

    def f(x):
        return A + (K - A) / (1 + np.exp(-B * (x - M)))

    return f

def five_param_logistic(p):
    """5p logistic function maker.

    Returns a function that accepts x and returns y for
    the 5-parameter logistic defined by p.

    The 4p logistic is defined by:
    y = A + (K - A) / ((1 + exp(-B*(x-M)))**E)

    Args:
        p: an iterable of length 5
            A, K, B, M, E = p

    Returns:
        A function that accepts a numpy array as an argument
        for x values and returns the y values for the defined 5pl curve.
    """
    A, K, B, M, E = p

    def f(x):
        return A + (K - A) / ((1 + np.exp(-B * (x - M)))**E)

    return f

def est_pstart(x, y):
    """basic estimation of a good place to start log likelihood maximization.

    Args:
        x: a numpy array of length n
            assumes a finite number of unique x values
        y: a numpy array of length n
            must be of dtype double or float so multiplication works

    Returns:
        p_start: an iterable of length 4 that should be a reasonable spot to
            start the optimization
            A, K, B, M = p_start
    """
    p_start = [0.01, 0.99, 0.2, 0]
    x_vals = np.unique(x)
    p_start[3] = np.mean(x_vals)
    y_est = np.array([np.mean(y[x == i]) for i in x_vals])
    midpoint_est = np.mean(np.where((y_est[0:-1] < 0.5) & (y_est[1:] >= 0.5)))
    if np.isnan(midpoint_est):
        return p_start
    p_start[3] = midpoint_est
    return p_start

def est_pstart_2(x, y):
    """basic estimation of a good place to start log likelihood maximization.

    Args:
        x: a numpy array of length n
            assumes a finite number of unique x values
        y: a numpy array of length n
            must be of dtype double or float so multiplication works

    Returns:
        p_start: an iterable of length 4 that should be a reasonable spot to
            start the optimization
            A, K, B, M = p_start
    """
    p_start = [0.2, 0]
    x_vals = np.unique(x)
    p_start[1] = np.mean(x_vals)
    y_est = np.array([np.mean(y[x == i]) for i in x_vals])
    midpoint_est = np.mean(np.where((y_est[0:-1] < 0.5) & (y_est[1:] >= 0.5)))
    if np.isnan(midpoint_est):
        return p_start
    p_start[1] = midpoint_est
    return p_start

def est_pstart_5(x, y):
    """basic estimation of a good place to start log likelihood maximization.

    Args:
        x: a numpy array of length n
            assumes a finite number of unique x values
        y: a numpy array of length n
            must be of dtype double or float so multiplication works

    Returns:
        p_start: an iterable of length 4 that should be a reasonable spot to
            start the optimization
            A, K, B, M = p_start
    """
    p_start = [0.01, 0.99, 0.2, 0, 1]
    x_vals = np.unique(x)
    p_start[3] = np.mean(x_vals)
    y_est = np.array([np.mean(y[x == i]) for i in x_vals])
    midpoint_est = np.mean(np.where((y_est[0:-1] < 0.5) & (y_est[1:] >= 0.5)))
    if np.isnan(midpoint_est):
        return p_start
    p_start[3] = midpoint_est
    return p_start

def ln_like(p, x, y):
    """log likelihood for fitting the four parameter logistic.

    Args:
        p: an iterable of length 4
            A, K, B, M = p
        x: a numpy array of length n
        y: a numpy array of length n
            must be of dtype double or float so multiplication works

    Returns:
        The log-likelihood that the samples y are drawn from a distribution
        where the 4pl(x; p) is the probability of getting y=1
    """
    p_4pl = four_param_logistic(p)
    probs = p_4pl(x)
    return np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))


def dln_like(p, x, y):
    """gradient of the log likelihood for fitting the four parameter logistic.

    Args:
        p: an iterable of length 4
            A, K, B, M = p
        x: a numpy array of length n
        y: a numpy array of length n
            must be of dtype double or float so multiplication works

    Returns:
        The gradient of the log-likelihood that the samples y are drawn from
        a distribution where the 4pl(x; p) is the probability of getting y=1
    """
    A, K, B, M = p

    def f(x):
        return A + (K - A) / (1 + np.exp(-B * (x - M)))

    def df(x):
        temp1 = np.exp(-B * (x - M))
        dK = 1.0 / (1.0 + temp1)
        dA = 1.0 - dK
        temp2 = temp1 / (1.0 + temp1) ** 2
        dB = (K - A) * (x - M) * temp2
        dM = -(K - A) * B * temp2
        return np.vstack((dA, dK, dB, dM))

    p_4pl = f(x)
    d_p_4pl = df(x)
    return np.sum(y * d_p_4pl / (p_4pl) - (1 - y) * d_p_4pl / (1 - p_4pl), 1)


def nll(*args):
    """negative log-likelihood for fitting the 4 param logistic."""
    return -ln_like(*args)


def ndll(*args):
    """negative grad of the log-likelihood for fitting the 4 param logistic."""
    return -dln_like(*args)

def ln_like_5(p, x, y):
    """log likelihood for fitting the four parameter logistic.

    Args:
        p: an iterable of length 5
            A, K, B, M, E = p
        x: a numpy array of length n
        y: a numpy array of length n
            must be of dtype double or float so multiplication works

    Returns:
        The log-likelihood that the samples y are drawn from a distribution
        where the 4pl(x; p) is the probability of getting y=1
    """
    p_5pl = five_param_logistic(p)
    probs = p_5pl(x)
    return np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))


def dln_like_5(p, x, y):
    """gradient of the log likelihood for fitting the four parameter logistic.

    Args:
        p: an iterable of length 5
            A, K, B, M, E = p
        x: a numpy array of length n
        y: a numpy array of length n
            must be of dtype double or float so multiplication works

    Returns:
        The gradient of the log-likelihood that the samples y are drawn from
        a distribution where the 5pl(x; p) is the probability of getting y=1
    """
    A, K, B, M, E = p

    def f(x):
        return A + (K - A) / ((1 + np.exp(-B * (x - M)))**E)

    def df(x):
        temp1 = np.exp(-B * (x - M))
        dK = 1.0 / ((1.0 + temp1)**E)
        dA = 1.0 - dK
        temp2 = temp1 / (1.0 + temp1) ** (E+1)
        dB = (K - A) * (x - M) * E * temp2
        dM = -(K - A) * B * E * temp2
        dE = -(K - A) * np.log(1.0 + temp1) / (1.0 + temp1) ** E
        return np.vstack((dA, dK, dB, dM, dE))
                                                                       
    p_5pl = f(x)
    d_p_5pl = df(x)
    return np.sum(y * d_p_5pl / (p_5pl) - (1 - y) * d_p_5pl / (1 - p_5pl), 1)


def nll_5(*args):
    """negative log-likelihood for fitting the 5 param logistic."""
    return -ln_like_5(*args)


def ndll_5(*args):
    """negative grad of the log-likelihood for fitting the 5 param logistic."""
    return -dln_like_5(*args)

def ln_like_2(p, x, y):
    """log likelihood for fitting the four parameter logistic.

    Args:
        p: an iterable of length 2
            B, M = p
        x: a numpy array of length n
        y: a numpy array of length n
            must be of dtype double or float so multiplication works

    Returns:
        The log-likelihood that the samples y are drawn from a distribution
        where the 4pl(x; p) is the probability of getting y=1
    """
    p_2pl = two_param_logistic(p)
    probs = p_2pl(x)
    return np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))


def dln_like_2(p, x, y):
    """gradient of the log likelihood for fitting the four parameter logistic.

    Args:
        p: an iterable of length 5
            B, M = p
        x: a numpy array of length n
        y: a numpy array of length n
            must be of dtype double or float so multiplication works

    Returns:
        The gradient of the log-likelihood that the samples y are drawn from
        a distribution where the 5pl(x; p) is the probability of getting y=1
    """
    B, M = p

    def f(x):
        return 1 / (1 + np.exp(-B * (x - M)))

    def df(x):
        temp1 = np.exp(-B * (x - M))
        temp2 = temp1 / (1.0 + temp1) ** 2
        dB = 1 * (x - M) * temp2
        dM = -1 * B * temp2
        return np.vstack((dB, dM))

    p_2pl = f(x)
    d_p_2pl = df(x)
    return np.sum(y * d_p_2pl / (p_2pl) - (1 - y) * d_p_2pl / (1 - p_2pl), 1)


def nll_2(*args):
    """negative log-likelihood for fitting the 5 param logistic."""
    return -ln_like_2(*args)


def ndll_2(*args):
    """negative grad of the log-likelihood for fitting the 5 param logistic."""
    return -dln_like_2(*args)