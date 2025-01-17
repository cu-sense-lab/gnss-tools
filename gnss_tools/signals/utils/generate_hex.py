
from ..gps.gps_l1ca import generate_code_sequence_L1CA
from ..gps_l2c import generate_code_sequence_L2CL, generate_code_sequence_L2CM
from ..gps.gps_l5 import generate_code_sequence_L5I, generate_code_sequence_L5Q

if __name__ == "__main__":

    # GPS L1CA
    out_filepath = "./hexfiles/gps_l1ca.txt"
    header = "GPS L1CA code sequences.  Second line contains space-separated PRN list.  Subsequent lines contain PRN codes in HEX format."
    prn_list = list(range(1, 33))
    for prn in prn_list:
        code_seq = generate_code_sequence_L1CA(prn)
        binstr = "".join(map(str, code_seq))
        hexstr = ""
        for i in range(len(binstr) // 8):
            hexstr += hex(int(binstr[i * 8:(i + 1) * 8], 2))[2:]
        remstr = binstr[i * 8:]
        remstr += "0" * (8 - len(remstr))
        hexstr += hex(int(remstr, 2))[2:]
        print(hexstr)
        break
    # GPS L2C

    # GPS L5