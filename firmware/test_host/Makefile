# Host tests for the hardware-independent firmware code (no ESP-IDF needed).
#   make -C firmware/test_host        build and run all tests
CC ?= gcc
ROOT := ../..
FW := ..
CFLAGS ?= -std=c99 -O1 -g -Wall -Wextra -Wshadow -Wno-unused-parameter -fsanitize=address,undefined
CFLAGS += -I$(ROOT)/core/include -I$(FW)/components/lepton/include -I$(FW)/components/bno085/include \
          -I$(FW)/components/audio/include -I$(FW)/components/buttons/include \
          -I$(FW)/components/detector/include -I$(FW)/components/power/include
LDLIBS = -lm

TESTS := test_lepton test_shtp test_audio_pack test_buttons test_detector_quant test_power_adc

all: run

test_lepton: test_lepton.c $(FW)/components/lepton/lepton_vospi_proto.c $(FW)/components/lepton/lepton_cci.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
test_shtp: test_shtp.c $(FW)/components/bno085/shtp_proto.c $(ROOT)/core/src/ps_nav.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
test_audio_pack: test_audio_pack.c $(FW)/components/audio/audio_pack.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
test_buttons: test_buttons.c $(FW)/components/buttons/button_fsm.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
test_detector_quant: test_detector_quant.c $(FW)/components/detector/detector_quant.c $(wildcard $(ROOT)/core/src/*.c)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
test_power_adc: test_power_adc.c $(FW)/components/power/power_calc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

build: $(TESTS)

out/audio/audio.bin: $(ROOT)/tools/make_audio_clips.py $(ROOT)/core/src/ps_alerts.c
	python3 $(ROOT)/tools/make_audio_clips.py --placeholder --out out/audio > /dev/null

run: build out/audio/audio.bin
	@set -e; for t in $(TESTS); do \
	  if [ $$t = test_audio_pack ]; then ./$$t out/audio/audio.bin; else ./$$t; fi; done

clean:
	rm -rf $(TESTS) out

.PHONY: all build run clean
