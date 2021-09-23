""" Task definitions for invoke command line utility for python bindings
    overview article.
"""
import invoke
import pathlib
import sys
import os
import shutil
import re
import glob

on_win = sys.platform.startswith("win")


@invoke.task
def clean(c):
    """Remove any built objects"""
    for file_pattern in (
        "*.o",
        "*.so",
        "*.obj",
        "*.dll",
        "*.exp",
        "*.lib",
    ):
        for file in glob.glob(file_pattern):
            os.remove(file)
    for dir_pattern in "Release":
        for dir in glob.glob(dir_pattern):
                shutil.rmtree(dir)


def print_banner(msg):
    print("==================================================")
    print("= {} ".format(msg))


@invoke.task()
def build_cmult(c, path=None):
    """Build the shared library for the sample C code"""
    # Moving this type hint into signature causes an error (???)
    c: invoke.Context
    if on_win:
        if not path:
            print("Path is missing")
        else:
            # Using c.cd didn't work with paths that have spaces :/
            path = f'"{path}vcvars32.bat" x86'  # Enter the VS venv
            path += f'&& cd "{os.getcwd()}"'  # Change to current dir
            path += "&& cl /LD cmult.c"  # Compile
            # Uncomment line below, to suppress stdout
            # path = path.replace("&&", " >nul &&") + " >nul"
            c.run(path)
    else:
        print_banner("Building C Library")
        cmd = "gcc -c -Wall -Werror -fpic cmult.c -I /usr/include/python3.7"
        invoke.run(cmd)
        invoke.run("gcc -shared -o libcmult.so cmult.o")
        print("* Complete")


@invoke.task()
def test_ctypes(c):
    """Run the script to test ctypes"""
    print_banner("Testing ctypes Module for C")
    # pty and python3 didn't work for me (win).
    if on_win:
        invoke.run("python ctypes_c_test.py")
    else:
        invoke.run("python3 ctypes_c_test.py", pty=True)


@invoke.task()
def test_ctypes_cpp(c):
    """Run the script to test ctypes"""
    print_banner("Testing ctypes Module for C++")
    # pty and python3 didn't work for me (win).
    if on_win:
        invoke.run("python ctypes_cpp_test.py")
    else:
        invoke.run("python3 ctypes_cpp_test.py", pty=True)

    """
        Having invoke run commands doesn't always behave exactly as it would, had we done it
        manually from the command line. The "pty=True" statement ensures things work as we
        expect it too in this manner. When a human interacts with a terminal the terminal includes
        extra colouring and typesetting to make the output more readable. This is not necessary
        if we are piping into another problem, since it only results in extra unnecessary 
        garbage. 
    """



@invoke.task()
def build_cppmult(c):
    """Build the shared library for the sample C++ code"""
    print_banner("Building C++ Library")
    invoke.run(
        "g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC cppmult.cpp "
        "-o libcppmult.so "
    )
    print("* Complete")

    """ 
        We must use the -shared flag to create a shared library (.so extension)
        on linux, or dynamic link library (.dll extension) on windows. This enables
        us to create libraries that can be shared between programs/executables. 
        Essentially this is smart because it saves memory. Programs can share the
        same source code.
        
        The -fPIC flag is there to ensure the code is position independent. Instead
        of working with absolute memory addresses, we work in relative terms, such as
        in the following pseudo-assembly code:
        
        Relative addresses:
        -------------------
        100: COMPARE REG1, REG2
        101: JUMP_IF_EQUAL CURRENT+10
        ...
        111: NOP

        Absolute addresses:
        -------------------
        100: COMPARE REG1, REG2
        101: JUMP_IF_EQUAL 111
        ...
        111: NOP
        
        This ensures the code works across platforms.
    """


@invoke.task(
    clean,
    build_cmult,
    build_cppmult,
    test_ctypes,
    test_ctypes_cpp,
    )
def all(c):
    """Build and run all tests"""
    pass
