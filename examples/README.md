# Examples 

This folder contains a list of example applications that can be done with Golum. To check out the GOLUM methodology, please see the [method paper](https://watermark.silverchair.com/stab1991.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAsowggLGBgkqhkiG9w0BBwagggK3MIICswIBADCCAqwGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMQQIXBH6lapOeaqA9AgEQgIICfeNsv55Kc6P7agvZeD-m1Dj6dpxokb_w8lOYJkkBsLm_C007h_KuUlI8vVy_H1e5ZUhwVrlqTVqdgUFsoyjRd6csSo7meHGwAESVG6D2eikjXsxqVk225-Qc-1EARbWR1dx3HvbKJflAhwCog0Dnsg3WHpHufT_1bAX0dGVBYedLmfc7r4b2ifRL-OqGVeIry8I8DBs1vjSc_ktX5ahsAeFMuwjyZ5Y9DMc4nfTbB2kYLQMQUb6gyGG887kBcGRtZVbIdpXUn_7UQDvHSMOiL4EPlATqHFvy1mgoIMiYJNxMnlsY8wSSp4BBdWMwGJyIfkbZfzmj6UoPRqXoi8Eazp85-tURVIbHHyV-NHSlVzcDtvTIMT3tfC6zICSEsPIYBTEfbs5n1aFWuJNCBmNrJ-a4YU9jWRp_YnDbAeJ-AS44CqvCY8rTGatnTFwOnwMHmyeBrSUyqFkcw05iKop5RGKtI-JyPFSSUmkrCP22AZVpt6GMgGDkeWdmYl4Ys_bSHmHMHc6ZzYgDaudaNIYBcU1o_kyiZU9crzT_QznfSosF0fljSg6eUYO1XJMJaZgJKl3DdvM39xMBVvSZYmwgNYbFt7ogZBGg3vqq691lkGHGhqK5gw8sACRgMR9XPZLqXWhaEmYhkfcMm7WCJ9Zfo3E3xtUjiUGpfhRae3hmsWa9VHTsDmSF7SauJAFRMz_rOIY8oOkiCEew7VQe77Bmp7VLU-GFXucD0DXLg7dizdsjqGeUVQbjLwr1pS23BUyNsGR-1vk913i8aM6lOvGRnEFDgEsqe7QLffwnGyvWSRYCwgJGZbW9HrrkrkiPC2XlXrytbUgHdXbKe0b453Y)

**NOTE:** The examples use the `pymultinest` sampler, which can be installed using `conda install -c conda-forge pymultinest`. If you do not want to do this, you can also change the sampler to `dynesty`, installed by default with `bilby`.

#### GOLUM Examples
- `Image1.py` Example file for the analysis of the reference image.
- `Image2.py` Example file for the analysis of the second image in the event pair. Here, we use the (mu_rel, delta_t, delta_n) parametrisation
- `Image2_DlTcN.py` Example analysis for the second image using a different parametrisation, hence (Dl2, tc2, n_2).
- `LensedInjection_UnlensedRecovery.py` Example analysis for an injected lensed waveform and an unlensed recovery. This is needed to compute the Coherence ratio.
- `FourImageAnalysis.py` Example analysis for a quadruply lensed event. This shows how Golum distributes the run to save time
- `ReweightResultLensingStat.py` Example file to reweigh the Golum results using some lensing statistics models present in Golum. We use the Mgal model from [More and More](https://arxiv.org/pdf/2111.03091.pdf) and the Rgal model from [Haris et al](https://arxiv.org/pdf/1807.07062.pdf)

A typical `GOLUM` analysis start by analyzing the first image (running the `Image1.py`) before analyzing the next images (either with the `Image2.py` or `FourImageAnalysis.py`). In order to avoid running away samples, it is suggested to use more samples for the more images (even if this decreases a bit the run time).