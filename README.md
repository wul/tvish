tvish
======

Traffic violation inquiry tool in Shanghai.
This tool need following Python package installed.
    
    NumPy
    Pillow
    scikit-learn    

Usage
======

    %tvish [-b]

It checks the traffic violation record in Shanghai Traffic Information Network(上海交通信息网 http://www.shjtaq.com). By examining the verification code via machine learning algorithms, it posts auto information and gets vilation results. '-b' parameter is used for building training examples.

Use command python3 tvish.py 2>/dev/null to filter out all warnings. 

Users should have a configuration file called ".tvish" under your home directory. 
Here is an example:

    [浙D18888]
    Type=02/小型汽车号牌
    EngineNumber=33193338

Users need only change the section name(浙D18888) and EngineerNumber. If two or more autos need to be checked, just add other sections.


