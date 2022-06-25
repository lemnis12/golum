import setuptools

setuptools.setup(
    name="golum",
    version="0.2.3",
    author="Justin Janquart, Haris K, Otto A. Hannuksela",
    author_email="j.janquart@uu.nl",
    description ="Gravitational-wave analysis Of Lensed and Unlensed waveform Models",
    url="https://github.com/lemnis12/golum",
    packages=["golum",
              "golum.PE",
              "golum.Population",
              "golum.Lookup",
              "golum.Tools"],
    package_data={"golum.Population" : ["StatisticsModels/out_fincat_AL/all_lensprop.txt",
                                        "StatisticsModels/out_fincat_AL/dblall_imgprop.txt",
                                        "StatisticsModels/out_fincat_AL/quadall_imgprop.txt",
                                        "StatisticsModels/out_fincat_Ap/all_lensprop.txt",
                                        "StatisticsModels/out_fincat_Ap/dblall_imgprop.txt",
                                        "StatisticsModels/out_fincat_Ap/quadall_imgprop.txt",
                                        "StatisticsModels/out_fincat_CE/all_lensprop.txt",
                                        "StatisticsModels/out_fincat_CE/dblall_imgprop.txt",
                                        "StatisticsModels/out_fincat_CE/quadall_imgprop.txt",
                                        "StatisticsModels/out_fincat_ET/all_lensprop.txt",
                                        "StatisticsModels/out_fincat_ET/dblall_imgprop.txt",
                                        "StatisticsModels/out_fincat_ET/quadall_imgprop.txt",
                                        "StatisticsModels/out_fincat_O3/all_lensprop.txt",
                                        "StatisticsModels/out_fincat_O3/dblall_imgprop.txt",
                                        "StatisticsModels/out_fincat_O3/quadall_imgprop.txt",
                                        "StatisticsModels/Rgal_dt.txt",
                                        "StatisticsModels/unlensedpairs/unlensedpairs_tdmag_AL.txt",
                                        "StatisticsModels/unlensedpairs/unlensedpairs_tdmag_Ap.txt",
                                        "StatisticsModels/unlensedpairs/unlensedpairs_tdmag_CE.txt",
                                        "StatisticsModels/unlensedpairs/unlensedpairs_tdmag_ET.txt",
                                        "StatisticsModels/unlensedpairs/unlensedpairs_tdmag_O3.txt",
                                        "StatisticsModels/unlensedpop/all_bbh_unlensprop_O3.txt",
                                        "StatisticsModels/unlensedpop/getcat_unlens4lenspop.txt",
                                        "StatisticsModels/unlensedpop/get_unlensedpairs.py"]},
    install_requires = ["scipy", 
                        "bilby",
                        "numpy"],
    classifiers=["Development Status :: 4 - Beta",
                 "Intended Audience :: Science/Research",
                 "Topic :: Scientific/Engineering :: Physics",
                 "Programming Language :: Python :: 3.8",
                 "Programming Language :: Python :: 3.9",
                 "License :: OSI Approved :: MIT License"],
    python_requires = ">=3.7"
    )