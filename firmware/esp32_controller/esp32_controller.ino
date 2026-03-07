/*
 * Autonomous Adaptive Resonator Control System
 * ESP32 Hardware-in-the-Loop Controller
 * * HARDWARE WIRING:
 * ----------------
 * AD9834 DDS (Waveform Generation) -> VSPI Bus
 * - FSYNC (CS) : Pin 5
 * - SCLK       : Pin 18
 * - SDATA (MOSI): Pin 23
 * * MCP3202 ADC (Amplitude Measurement) -> HSPI Bus
 * - CS         : Pin 15
 * - CLK        : Pin 14
 * - DIN (MOSI) : Pin 13
 * - DOUT (MISO): Pin 12
 * * ENVELOPE DETECTOR HARDWARE (External):
 * Resonator Output -> OPA2134 Precision Rectifier -> 0.1uF Hold Cap ->
 * RC Low-Pass Filter (R=33k, C=100nF, fc≈48Hz) -> MCP3202 ADC CH0
 */

#include <SPI.h>

// --- PIN DEFINITIONS ---
#define AD9834_CS_PIN 5
#define ADC_CS_PIN 15

// --- TIMING & OVERSAMPLING CONSTANTS ---
#define SETTLE_DELAY_US                                                        \
  200 // Delay for envelope detector to settle after freq update
#define OVERSAMPLE_COUNT 64      // Number of ADC reads to average
#define WATCHDOG_TIMEOUT_MS 5000 // Halt if Python disconnects

// --- COMPILE FLAGS ---
// Uncomment to print phase 1 & 2 microsecond timing data
// #define DEBUG_MODE

// --- SPI OBJECTS ---
SPIClass vspi(VSPI);
SPIClass hspi(HSPI);

unsigned long lastCommandTime = 0;
float currentFrequency = 500000.0;

void setup() {
  Serial.begin(921600);

  // Initialize VSPI for AD9834
  pinMode(AD9834_CS_PIN, OUTPUT);
  digitalWrite(AD9834_CS_PIN, HIGH);
  vspi.begin(18, 19, 23, AD9834_CS_PIN); // SCLK, MISO (unused), MOSI, SS

  // Initialize HSPI for MCP3202
  pinMode(ADC_CS_PIN, OUTPUT);
  digitalWrite(ADC_CS_PIN, HIGH);
  hspi.begin(14, 12, 13, ADC_CS_PIN); // SCLK, MISO, MOSI, SS

  // AD9834 initialization (reset, set B28 for two consecutive writes)
  writeAD9834(0x2100);
  setFrequency(currentFrequency);
  writeAD9834(0x2000); // clear reset

  lastCommandTime = millis();
}

void loop() {
  // Watchdog: Hold operations if Python script drops
  if (millis() - lastCommandTime > WATCHDOG_TIMEOUT_MS) {
    // Failsafe state
    return;
  }

  // Parse Serial Commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command.startsWith("SET_FREQ")) {
      float targetFreq = command.substring(9).toFloat();

      // Execute strict two-phase control cycle
      float amplitude = runControlCycle(targetFreq);

      // Respond to Python
      Serial.print("MEASURE ");
      Serial.println(amplitude, 4);

      lastCommandTime = millis();
    }
  }
}

// ---------------------------------------------------------
// PHASE CONTROL LOOP (F-008 Fix: SPI Arbitration)
// ---------------------------------------------------------
float runControlCycle(float freq_hz) {
#ifdef DEBUG_MODE
  unsigned long t_start = micros();
#endif

  // PHASE 1: Excitation (VSPI Active, HSPI Idle)
  setFrequency(freq_hz);

#ifdef DEBUG_MODE
  unsigned long t_phase1 = micros();
#endif

  // Hardware Delay: Wait for envelope detector RC circuit to settle
  delayMicroseconds(SETTLE_DELAY_US);

  // PHASE 2: Measurement (HSPI Active, VSPI Idle)
  float amp = measureAmplitude();

#ifdef DEBUG_MODE
  unsigned long t_phase2 = micros();
  Serial.print("Phase 1 us: ");
  Serial.print(t_phase1 - t_start);
  Serial.print(" | Phase 2 us: ");
  Serial.println(t_phase2 - t_phase1);
#endif

  return amp;
}

// ---------------------------------------------------------
// AD9834 DDS WAVEFORM GENERATION (F-006 Fix)
// ---------------------------------------------------------
void writeAD9834(uint16_t data) {
  vspi.beginTransaction(SPISettings(20000000, MSBFIRST, SPI_MODE2));
  digitalWrite(AD9834_CS_PIN, LOW);
  vspi.transfer16(data);
  digitalWrite(AD9834_CS_PIN, HIGH);
  vspi.endTransaction();
}

void setFrequency(float freq_hz) {
  // Frequency Word = (Desired Freq / Master Clock) * 2^28
  // Assuming standard 75MHz oscillator on AD9834 module
  uint32_t freq_word = (uint32_t)((freq_hz / 75000000.0) * 268435456.0);

  // Split 28-bit word into two 14-bit words for FREQ0 register (address 0x4000)
  uint16_t LSB = 0x4000 | (freq_word & 0x3FFF);
  uint16_t MSB = 0x4000 | ((freq_word >> 14) & 0x3FFF);

  // setPhaseContinuous: Write continuously without resetting phase accumulator
  writeAD9834(LSB);
  writeAD9834(MSB);
}

// ---------------------------------------------------------
// MCP3202 ADC ENVELOPE MEASUREMENT (F-007 Fix)
// ---------------------------------------------------------
float measureAmplitude() {
  uint32_t accumulator = 0;

  hspi.beginTransaction(SPISettings(2000000, MSBFIRST, SPI_MODE0));

  for (int i = 0; i < OVERSAMPLE_COUNT; i++) {
    digitalWrite(ADC_CS_PIN, LOW);

    // MCP3202 CH0 read format
    hspi.transfer(0x01);               // Start bit
    uint8_t msb = hspi.transfer(0xA0); // SGL/DIFF, ODD/SIGN, MSBF
    uint8_t lsb = hspi.transfer(0x00);

    digitalWrite(ADC_CS_PIN, HIGH);

    // Extract 12-bit result
    uint16_t result = ((msb & 0x0F) << 8) | lsb;
    accumulator += result;
  }

  hspi.endTransaction();

  // Average and normalize to 0.0 - 1.2 float (matching python sim bounds)
  // Assuming 3.3V Vref maxes at 4095
  float average_adc = (float)accumulator / OVERSAMPLE_COUNT;
  float normalized_amp = (average_adc / 4095.0) * 1.2;

  return normalized_amp;
}