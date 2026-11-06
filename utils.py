sessions = {
    "003_TKA": {"00_01": ["01"], "02_04": ["01"], "08": ["01"]},
    "006_PKA":{"00_01":["01","02","03"], "02_04":["01"],"05_07":["01"], "08":[str(i).zfill(2) for i in range(1, 8)]},
    "007_TKA": {"02_04": ["01"], "05_07": ["01"], "08": ["01", "02"]},
    "011_TKA": {"00_01": ["01","02"], "08": [str(i).zfill(2) for i in range(1, 4)]},
    "012_1_PKA": {"00_01": [str(i).zfill(2) for i in range(1, 5)]},
    "012_2_PKA": {"05_07": ["01","02"]},
    "015_PKA": {"00_01":["01","02"]},
    "016_PKA": {"00_01":["01"]},
    "017_PKA": {"00_01":["01"]},
    "020_PKA": {"00_01":["01"]},
    "022_PKA": {"00_01":["01"]},
    "027_PKA": {"02_04":["01"]},
    "031_PKA": {"02_04":["01"]},
    "032_PKA": {"00_01": ["01"]},
    "033_PKA": {"00_01": ["01"]},
    "035_PKA": {"00_01": [str(i).zfill(2) for i in range(1, 4)],
                "02_04": ["01"], "08": [str(i).zfill(2) for i in range(1, 7)]},
    "036_PKA": {"00_01": [str(i).zfill(2) for i in range(1, 4)],
               "02_04": ["01"], "05_07": ["01"], "08": [str(i).zfill(2) for i in range(1, 9)]},
    #"037_TKA":{"00_01":[str(i).zfill(2) for i in range(1, 4)]}
    "037_TKA": {"00_01": [str(i).zfill(2) for i in range(1, 4)],
                "02_04": ["01"], "05_07": [str(i).zfill(2) for i in range(1, 9)]},
    "038_TKA": {"05_07": ["01"]}
}

root_path = r"C:\Users\marco\Desktop\Resume_transcription\Labeling"

segment = "15"

label_order = ["Communication","Coordination","Cooperation","Leadership","Monitoring"]
