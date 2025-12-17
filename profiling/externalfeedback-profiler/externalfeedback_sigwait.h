#ifndef EXTERNALFEEDBACK_SIGWAIT_H
#define EXTERNALFEEDBACK_SIGWAIT_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize external feedback system with sigwait pattern
// verbose: 1 to enable debug output, 0 for quiet
// Returns: 0 on success, -1 on failure
int externalfeedback_init(int verbose);

// Shutdown external feedback system
void externalfeedback_shutdown(void);

// Trigger handshake programmatically (for testing without signals)
void externalfeedback_trigger_hello(void);

#ifdef __cplusplus
}
#endif

#endif // EXTERNALFEEDBACK_SIGWAIT_H
