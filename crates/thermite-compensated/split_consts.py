import decimal
import struct

# Set precision high enough to capture the tail accurately
decimal.getcontext().prec = 100

def split(val_str, parts=2, is_f32=False):
    """
    Splits a high-precision string/decimal into `parts` non-overlapping values.
    Returns a tuple of python floats.
    """
    if isinstance(val_str, decimal.Decimal):
        d = val_str
    else:
        d = decimal.Decimal(val_str)

    results = []
    for _ in range(parts):
        if is_f32:
            # Round to f32 precision via struct round-trip
            # struct.pack/unpack ensures the value is exactly representable in 32 bits
            f_bytes = struct.pack('f', float(d))
            val = struct.unpack('f', f_bytes)[0]

            # Convert back to Decimal for exact subtraction
            # Note: The python float 'val' is an f64, but it holds an f32 value.
            # Converting it to Decimal preserves that exact value.
            val_decimal = decimal.Decimal(float(val))
        else:
            # Native Python float is f64
            val = float(d)
            val_decimal = decimal.Decimal(val)

        results.append(val)

        # Update the remainder
        d -= val_decimal

    return tuple(results)

def print_f64_result(name, val_str):
    hi64, lo64 = split(val_str, parts=2, is_f32=False)
    print(f"{name} = (\"{hi64.hex()}\", \"{lo64.hex()}\"),")

def print_f64_triple_result(name, val_str):
    hi, mid, lo = split(val_str, parts=3, is_f32=False)
    print(f"{name} = (\"{hi.hex()}\", \"{mid.hex()}\", \"{lo.hex()}\"),")

def print_f32_result(name, val_str):
    hi32, lo32 = split(val_str, parts=2, is_f32=True)
    print(f"{name} = (\"{float(hi32).hex()}\", \"{float(lo32).hex()}\"),")

def print_f32_triple_result(name, val_str):
    hi, mid, lo = split(val_str, parts=3, is_f32=True)
    print(f"{name} = (\"{float(hi).hex()}\", \"{float(mid).hex()}\", \"{float(lo).hex()}\"),")

if __name__ == "__main__":
    consts = {
        "NEG_ZERO": "-0.0",
        "E": "2.7182818284590452353602874713526624977572470937000",
        "EGAMMA": "0.57721566490153286060651209008240243104215933593992",
        "FRAC_1_PI": "0.31830988618379067153776752674502872406891929148091",
        "FRAC_1_SQRT_2": "0.70710678118654752440084436210484903928483593768847",
        "FRAC_1_SQRT_3": "0.57735026918962576450914878050195745564760175127013",
        "FRAC_2_PI": "0.63661977236758134307553505349005744813783858296183",
        "FRAC_1_SQRT_PI": "0.56418958354775628694807945156077258584405062932900",
        "FRAC_2_SQRT_PI": "1.1283791670955125738961589031215451716881012586580",
        "FRAC_SQRT_PI_2": "0.88622692545275801364908374167057259139877472806119",
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
        "PI_SQUARED": "9.8696044010893586188344909998761511353136994072408",
        "PI_CUBED": "31.006276680299820175476315067101395202225288565885",
        "PI_TESSERACTED": "97.409091034002437236440332688705111249727585672685",
        "SQRT_2": "1.4142135623730950488016887242096980785696718753769",
        "SQRT_3": "1.7320508075688772935274463415058723669428052538104",
        "SQRT_E": "1.6487212707001281468486507878141635716537761007101",
        "TAU": "6.2831853071795864769252867665590057683943387987502",
        "SQRT_FRAC_PI_2": "0.79788456080286535587989211986876373695171726232987",
        "SQRT_2_PI": "2.5066282746310005024157652848110452530069867406099",
        "PHI": "1.6180339887498948482045868343656381177203091798058",
        "FRAC_1_3": "0.33333333333333333333333333333333333333333333333333333333333333",
        "FRAC_1_6": "0.16666666666666666666666666666666666666666666666666666666666667",
    }

    f64_consts = {
        "EPSILON": "0.00000000000000000000000000000002465190328815662",
        "SQRT_EPSILON": "0.0000000000000001570092458683775",
        "FOURTH_ROOT_EPSILON": "0.000000012530333031024255",
    }

    f32_consts = {
        "EPSILON": "0.000000000000007105427411152052",
        "SQRT_EPSILON": "0.0000000842936973394337",
        "FOURTH_ROOT_EPSILON": "0.0002903337688582464",
    }

    print("---- f64 split ----")
    for name, val_str in consts.items():
        print_f64_result(name, val_str)
    for name, val_str in f64_consts.items():
        print_f64_result(name, val_str)

    print("\n---- f32 split ----")
    for name, val_str in consts.items():
        print_f32_result(name, val_str)
    for name, val_str in f32_consts.items():
        print_f32_result(name, val_str)

    # Calculate 1/ln(n) for 3 <= n <= 32
    inv_logs = []
    for n in range(3, 33):
        # decimal.Decimal(n).ln() is high precision ln(n)
        val = decimal.Decimal(1) / decimal.Decimal(n).ln()
        inv_logs.append(val)

    print("\n---- 1/ln(n) f64 [3..32] ----")
    print("[")
    for val in inv_logs:
        hi, lo = split(val, parts=2, is_f32=False)
        print(f"    (\"{hi.hex()}\", \"{lo.hex()}\"),")
    print("]")

    print("\n---- 1/ln(n) f32 [3..32] ----")
    print("[")
    for val in inv_logs:
        hi, lo = split(val, parts=2, is_f32=True)
        print(f"    (\"{float(hi).hex()}\", \"{float(lo).hex()}\"),")
    print("]")

    print("\n---- LN_2 Triple Split (Calculated) ----")
    # We calculate ln(2) fresh to ensure enough digits for the 3rd term
    ln2_calc = decimal.Decimal(2).ln()
    print_f64_triple_result("LN_2_F64", ln2_calc)
    print_f32_triple_result("LN_2_F32", ln2_calc)