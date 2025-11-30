import decimal
import struct

# Set precision high enough to capture the tail accurately
decimal.getcontext().prec = 100

def split_f64(val_str):
    """Splits a high-precision string into two f64s (hi, lo)."""
    d = decimal.Decimal(val_str)

    # 1. Cast to native float (f64)
    hi = float(d)

    # 2. Subtract high from original to get exact remainder
    #    Then cast remainder to f64
    lo = float(d - decimal.Decimal(hi))

    return hi, lo

def split_f32(val_str):
    """Splits a high-precision string into two f32s (hi, lo)."""
    d = decimal.Decimal(val_str)

    # 1. Cast to f32 (via struct pack/unpack to force 32-bit rounding)
    #    Python floats are f64, so we must round-trip through bytes.
    hi_bytes = struct.pack('f', float(d))
    hi = struct.unpack('f', hi_bytes)[0]

    # 2. Subtract the EXACT f32 value from the original decimal
    #    Note: We convert 'hi' (which is an f32 value stored in an f64)
    #    back to Decimal to perform the subtraction.
    rem_d = d - decimal.Decimal(float(hi))

    # 3. Cast remainder to f32
    lo_bytes = struct.pack('f', float(rem_d))
    lo = struct.unpack('f', lo_bytes)[0]

    return hi, lo

def print_f64_result(name, val_str):
    hi64, lo64 = split_f64(val_str)

    print(f"{name} = (\"{hi64.hex()}\", \"{lo64.hex()}\"),")

def print_f32_result(name, val_str):
    hi32, lo32 = split_f32(val_str)

    print(f"{name} = (\"{float(hi32).hex()}\", \"{float(lo32).hex()}\"),")

if __name__ == "__main__":
    consts = {
        "ZERO": "0.0",
        "ONE": "1.0",
        "E": "2.7182818284590452353602874713526624977572470937000",
        "EGAMMA": "0.57721566490153286060651209008240243104215933593992",
        "FRAC_1_PI": "0.31830988618379067153776752674502872406891929148091",
        "FRAC_1_SQRT_2": "0.70710678118654752440084436210484903928483593768847",
        "FRAC_1_SQRT_3": "0.57735026918962576450914878050195745564760175127013",
        "FRAC_2_PI": "0.63661977236758134307553505349005744813783858296183",
        "FRAC_1_SQRT_PI": "0.56418958354775628694807945156077258584405062932900",
        "FRAC_2_SQRT_PI": "1.1283791670955125738961589031215451716881012586580",
        "FRAC_1_SQRT_TAU": "0.39894228040143267793994605993438186847585863116493",
        "FRAC_PI_2": "1.5707963267948966192313216916397514420985846996876",
        "FRAC_PI_3": "1.0471975511965977461542144610931676280657231331250",
        "FRAC_PI_4": "0.78539816339744830961566084581987572104929234984378",
        "FRAC_PI_6": "0.52359877559829887307710723054658381403286156656252",
        "FRAC_PI_8": "0.39269908169872415480783042290993786052464617492189",
        "FRAC_PI_180": "0.017453292519943295769236907684886127134428718885417",
        "FRAC_180_PI": "57.295779513082320876798154814105170332405472466564",
        "LN_2": "0.69314718055994530941723212145817656807550013436026",
        "LN_10": "2.3025850929940456840179914546843642076011014886288",
        "LN_PI": "1.1447298858494001741434273513530587116472948129153",
        "FRAC_LN_PI_2": "0.57236494292470008707171367567652935582364740645766",
        "LOG2_10": "3.3219280948873623478703194294893901758648313930246",
        "LOG2_E": "1.4426950408889634073599246810018921374266459541530",
        "LOG10_2": "0.30102999566398119521373889472449302676818988146211",
        "LOG10_E": "0.43429448190325182765112891891660508229439700580367",
        "PI": "3.1415926535897932384626433832795028841971693993751",
        "SQRT_2": "1.4142135623730950488016887242096980785696718753769",
        "SQRT_3": "1.7320508075688772935274463415058723669428052538104",
        "SQRT_E": "1.6487212707001281468486507878141635716537761007101",
        "TAU": "6.2831853071795864769252867665590057683943387987502",
        "SQRT_FRAC_PI_2": "0.79788456080286535587989211986876373695171726232987",
        "SQRT_2_PI": "2.5066282746310005024157652848110452530069867406099",
        "PHI": "1.6180339887498948482045868343656381177203091798058",
    }

    f64_consts = {
        "EPSILON": "0.00000000000000000000000000000002465190328815661891911651766508706968",
        "SQRT_EPSILON": "0.00000000000000000000000000000002465190328815661891911651766508706968",
        "FOURTH_ROOT_EPSILON": "0.00000000000000000000000000000002465190328815661891911651766508706968",
    }

    f32_consts = {
        "EPSILON": "0.000000000000007105427357601002",
        "SQRT_EPSILON": "0.00000008429369702178806",
        "FOURTH_ROOT_EPSILON": "0.00029033376831121120",
    }

    print("---- f64 split ----")
    for name, val_str in consts.items():
        print_f64_result(name, val_str)
    for name, val_str in f64_consts.items():
        print_f64_result(name, val_str)

    print("---- f32 split ----")
    for name, val_str in consts.items():
        print_f32_result(name, val_str)
    for name, val_str in f32_consts.items():
        print_f32_result(name, val_str)