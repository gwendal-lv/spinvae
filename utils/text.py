import os
import sys
from contextlib import contextmanager, redirect_stdout


# from https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
#    also works with prints from external C shared libs (such as renderman)
@contextmanager
def hidden_prints(to=os.devnull, filter_stderr=False):
    """
    :param to:
    :param filter_stderr: If True, filters std::cerr instead of std::cout
    """
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = (sys.stderr.fileno() if filter_stderr else sys.stdout.fileno())

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_std_stream(to, filter_stderr: bool):
        if filter_stderr:
            sys.stderr.close()  # + implicit flush()
        else:
            sys.stdout.close()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        if filter_stderr:
            sys.stderr = os.fdopen(fd, 'w')  # Python writes to fd
        else:
            sys.stdout = os.fdopen(fd, 'w')

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_std_stream(to=file, filter_stderr=filter_stderr)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            # restore stdout. buffering and flags such as CLOEXEC may be different
            _redirect_std_stream(to=old_stdout, filter_stderr=filter_stderr)


# from https://stackoverflow.com/questions/71024919/how-to-capture-prints-in-real-time-from-function
class StdOutPrefixAdder:

    def __init__(self, prefix: str):
        self.buf = ""
        self.prefix = prefix
        self.real_stdout = sys.stdout

    def write(self, buf):
        # emit on each newline
        while buf:
            try:
                newline_index = buf.index("\n")
            except ValueError:
                # no newline, buffer for next call
                self.buf += buf
                break
            # get data to next newline and combine with any buffered data
            data = self.buf + buf[:newline_index + 1]
            self.buf = ""
            buf = buf[newline_index + 1:]
            # perform complex calculations... or just print with a note.
            self.real_stdout.write(self.prefix + data)

