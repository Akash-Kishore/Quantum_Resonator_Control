# ARCS Project — Chat Restoration Prompt

## Context
I am a DSCE (Dayananda Sagar College of Engineering) Electronics & Telecommunication
Engineering student working on a **Mini Project (Sem 4)** called the
**Autonomous Resonator Control System (ARCS)** — V4 AUTHORITATIVE build (2025-26).

The project uses a reinforcement learning agent running on an ESP32 to autonomously
track and maintain the resonance frequency of an LC tank circuit at ~556 kHz by
controlling a varactor diode bias and reading amplitude feedback through an ADC.

---

## Hardware Stack

| Component | Part | Package | Bus / GPIO |
|---|---|---|---|
| Microcontroller | ESP32 DevKit V1 | 30-pin | — |
| DAC | MCP4921 12-bit | DIP-8 | VSPI: SCK=18, MOSI=23, CS=5 |
| ADC | MCP3202 12-bit dual-ch | DIP-8 | HSPI: SCK=14, MISO=12, MOSI=13, CS=15 |
| Op-Amp (buffer) | OPA2134UA | SOIC-8 → DIP-8 adapter | Analog rail only |
| Varactor | BB909 or MV209 | TO-92 | Bias via GPIO25 PWM RC filter |
| Inductor | Bourns SRR1280-100M | SMD (berg pin adapter) | 10µH, DCR ~0.5Ω |
| Power Reg ×2 | AMS1117-3.3 | SOT-223 | Analog + Digital rails separate |
| Schottky Diode | BAT54-13-F | SOT-23 (0805 adapter) | Envelope rectifier |

**Key net labels:** `ANALOG_3V3`, `DIGITAL_3V3`, `GND`, `VSPI_SCK/MOSI/CS_DAC`,
`HSPI_SCK/MISO/MOSI/CS_ADC`, `DAC_OUT`, `RESONATOR_IN`, `VARACTOR_BIAS`,
`PWM_DRIFT` (GPIO25), `DETECTOR_IN`, `RECT_OUT`, `LPF_OUT`, `ENVELOPE_OUT`

---

## Five Circuit Stages

### Stage 1 — AMS1117 Power Supply
- Two separate AMS1117-3.3 regulators: U1 (ANALOG_3V3), U2 (DIGITAL_3V3)
- Each has 10µF electrolytic + 100nF X7R decoupling at output
- **LTspice result: V(n002) = 3.30073V ✅ DONE**

### Stage 2 — MCP4921 DAC + ESP32 VSPI
- MCP4921 LDAC tied permanently to GND
- VREF = DIGITAL_3V3, config word = 0x3000 | 12-bit code
- Output: VOUT → 200Ω (R1, series) → 100nF C6 (series, AC coupling) → RESONATOR_IN
- PWM drift: GPIO25 → 10kΩ (R2, series) → 1µF C7 (parallel to GND) → VARACTOR_BIAS
- **Wokwi simulation: ✅ DONE**

### Stage 3 — LC Resonator + Varactor Tuning
- L1: 10µH Bourns SRR1280 (Rser=0.5Ω in sim)
- C8: 8.2nF C0G (parallel with L1, to GND) — forms LC tank at ~556 kHz
- Varactor coupling: RESONATOR_IN → C10 100pF (series) → BB909 cathode
- BB909 anode → R3 100kΩ (series) → VARACTOR_BIAS
- Output tap: RESONATOR_IN → R4 10kΩ (series) → DETECTOR_IN
- **LTspice AC sweep result: f0 = 555.61 kHz, Lorentzian peak confirmed ✅ DONE**

### Stage 4 — Envelope Detector
- D2 BAT54: anode=DETECTOR_IN, cathode=RECT_OUT (series rectifier)
- C_HOLD 100nF **film** cap: RECT_OUT to GND (parallel, peak hold)
- R_LP 33kΩ ±1%: RECT_OUT → LPF_OUT (series)
- C_LPF 100nF **film or C0G ONLY** (NO X7R): LPF_OUT to GND (parallel)
  - fc = 1/(2π×33k×100n) ≈ 48 Hz
- OPA2134 unity-gain follower:
  - IN+ → LPF_OUT
  - IN− → short wire directly to OUT (unity-gain feedback — CRITICAL)
  - OUT → ENVELOPE_OUT
  - V+ → ANALOG_3V3 (not digital rail)
  - V− → GND
  - V1 (3.3V supply) connects ONLY to V+ and GND — isolated from signal path
- Expected output: ~1.25V DC at resonance (1.5V RF − 0.25V BAT54 Vf)
- **LTspice transient: ✅ DONE** (note: persistent staircase issue in sim due
  to LT1057 open-loop; real hardware with OPA2134 will be correct)

### Stage 5 — MCP3202 ADC + ESP32 HSPI
- CH0 = ENVELOPE_OUT, CH1 tied to GND via 10kΩ
- VDD and VREF both on ANALOG_3V3 rail
- 3-byte SPI protocol: send 0x01, 0x80, 0x00; result = ((byte2 & 0x0F)<<8)|byte3
- 64-sample averaging; normalise to [0.0, 1.0]; print `MEASURE %.6f`
- **Wokwi simulation: ✅ DONE**

---

## Simulation Tools Used

| Tool | Purpose |
|---|---|
| LTspice XVII | Stages 1, 3, 4 (analog SPICE simulation) |
| Wokwi (wokwi.com) | Stages 2, 5 (ESP32 firmware + SPI verification) |
| Proteus 8.5 | **Replaced by Wokwi** — not used |

---

## Key LTspice Notes

- **Diode model**: Use generic diode (press `D`), set `Value = BAT54_MOD`,
  add directive: `.model BAT54_MOD D(Is=1e-6 N=1.05 Vj=0.35 M=0.45 Rs=5)`
  The Toshiba TBAT54 component (Prefix=X) causes "sub-circuit not defined" error.
- **OPA2134 in LTspice**: Use LT1057 as substitute. V+ supply (V1=3.3V) connects
  ONLY to V+ pin and GND — not in signal path.
- **Unity-gain feedback wire**: OUT pin → IN− pin (short direct wire). Without
  this the op-amp is open-loop and output staircases at ~1.033V instead of 1.25V.

---

## Component Status (Ordered/Verified)

### Correct ✅
- MCP4921 DIP-8, MCP3202 DIP-8, AMS1117-3.3 ×2, OPA2134UA (on SOIC-8 adapter)
- MB102 830-point breadboard, Rubycon 10µF ×4, BAT54-13-F ×3
- YAGEO 10kΩ ±1% ×5, YAGEO 33kΩ ±1% ×2, YAGEO 100kΩ ±1% ×3
- YAGEO 1kΩ ±1% ×5, MF25 200Ω ±1%, SRR1280 10µH (SMD, needs berg pins)
- Murata 8.2nF C0G 0805, 100pF C0G 0805 ×2, 100nF 630V polyester film cap
- SOIC-8 to DIP-8 adapter (pack of 5), Berg strip 1×40 ×2

### Still Needed ❌
- **BB909 or MV209 varactor diode** (BB910 is wrong part — different C-V curve)
- **1µF capacitors ×3** for PWM RC filter (previous order was out of stock)

### Cautions ⚠
- X7R 100nF caps (TCC1206): use ONLY for decoupling, NOT for LP filter position
- SRR1280 inductor is SMD — solder berg strip pins to pads before breadboard use
- All 0805 SMD caps need berg strip leads or 0805-to-DIP adapters before use
- OPA2134UA must be soldered onto SOIC-8 adapter before inserting in breadboard

---

## Documents Already Generated

1. `Hardware_Integration_Plan_AUTHORITATIVE.docx` — original authoritative spec
2. `ARCS_Hardware_Build_Guide.docx` — component verification V1 + Proteus guide
3. `ARCS_Complete_Build_Guide_V2.docx` — full verification + breadboard + Wokwi
4. `ARCS_Sim_Guide_V3.docx` — LTspice fix guide + Wokwi replacing Proteus
5. `ARCS_DAC_ADC_Integration_Guide.docx` — **final doc**: Wokwi DAC/ADC sim +
   combined firmware + full 5-stage breadboard integration guide

---

## Breadboard Build Order (Do Not Skip Steps)

1. Build Stage 1 → verify both rails at 3.28–3.32V
2. Build Stage 2 → verify DAC sine on scope, SPI frames on logic analyser
3. Build Stage 3 → verify Lorentzian peak near 556 kHz, varactor shifts f0
4. Build Stage 4 → verify ENVELOPE_OUT ~3.0–3.2V at f0, drops off-resonance
5. Build Stage 5 → verify MEASURE ~1.0 at f0, ~0.5 off-resonance, StdDev < 0.01

---

## Current Status

All 5 simulations are complete. Waiting on BB909/MV209 varactor and 1µF caps
before starting physical breadboard construction. Next step after components
arrive: Stage 1 power supply build and verification.
