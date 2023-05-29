import setuptools

setuptools.setup(
    name="golum",
    version="1.0.0",
    author="Justin Janquart, Haris K, Otto A. Hannuksela",
    author_email="j.janquart@uu.nl",
    description="Gravitational-wave analysis Of Lensed and Unlensed Waveform Models",
    url="https://github.com/lemnis12/golum",
    packages=["golum",
              "golum.pe",
              "golum.population",
              "golum.lookup",
              "golum.tools"],
    package_data={"golum.population" : ["statistics_models/out_fincat_AL/all_lensprop.txt",
                                        "statistics_models/out_fincat_AL/dblall_imgprop.txt",
                                        "statistics_models/out_fincat_AL/quadall_imgprop.txt",
                                        "statistics_models/out_fincat_Ap/all_lensprop.txt",
                                        "statistics_models/out_fincat_Ap/dblall_imgprop.txt",
                                        "statistics_models/out_fincat_Ap/quadall_imgprop.txt",
                                        "statistics_models/out_fincat_CE/all_lensprop.txt",
                                        "statistics_models/out_fincat_CE/dblall_imgprop.txt",
                                        "statistics_models/out_fincat_CE/quadall_imgprop.txt",
                                        "statistics_models/out_fincat_ET/all_lensprop.txt",
                                        "statistics_models/out_fincat_ET/dblall_imgprop.txt",
                                        "statistics_models/out_fincat_ET/quadall_imgprop.txt",
                                        "statistics_models/out_fincat_O3/all_lensprop.txt",
                                        "statistics_models/out_fincat_O3/dblall_imgprop.txt",
                                        "statistics_models/out_fincat_O3/quadall_imgprop.txt",
                                        "statistics_models/Rgal_dt.txt",
                                        "statistics_models/unlensedpairs/unlensedpairs_tdmag_AL.txt",
                                        "statistics_models/unlensedpairs/unlensedpairs_tdmag_Ap.txt",
                                        "statistics_models/unlensedpairs/unlensedpairs_tdmag_CE.txt",
                                        "statistics_models/unlensedpairs/unlensedpairs_tdmag_ET.txt",
                                        "statistics_models/unlensedpairs/unlensedpairs_tdmag_O3.txt",
                                        "statistics_models/unlensedpop/all_bbh_unlensprop_O3.txt",
                                        "statistics_models/unlensedpop/getcat_unlens4lenspop.txt",
                                        "statistics_models/unlensedpop/get_unlensedpairs.py"]},
    install_requires = ["scipy", 
                        "bilby",
                        "numpy"],
    classifiers=["Development Status :: 5 - Production/Stable",
                 "Intended Audience :: Science/Research",
                 "Topic :: Scientific/Engineering :: Physics",
                 "Programming Language :: Python :: 3.8",
                 "Programming Language :: Python :: 3.9",
                 "License :: OSI Approved :: MIT License"],
    python_requires = ">=3.7"
    )