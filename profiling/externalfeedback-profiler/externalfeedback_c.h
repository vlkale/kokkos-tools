//@HEADER
// Pure C external feedback library header
//@HEADER

#ifndef EXTERNALFEEDBACK_C_H
#define EXTERNALFEEDBACK_C_H

#ifdef __cplusplus
extern "C" {
#endif

void externalfeedback_init(int verbose, int disable_daemon, unsigned int interval_ms);
void externalfeedback_finalize(void);
void externalfeedback_trigger_hello(void);

#ifdef __cplusplus
}
#endif

#endif
