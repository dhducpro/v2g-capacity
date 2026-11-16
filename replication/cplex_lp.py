"""Solve sparse linear programs with the IBM ILOG CPLEX Callable Library."""
import ctypes as ct
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse

ptr = ct.c_void_p
integer = ct.c_int
number = ct.c_double
integers = ct.POINTER(integer)
numbers = ct.POINTER(number)


def library():
    location = os.environ.get("CPLEX_LIBRARY")
    if not location:
        exe = os.environ.get("CPLEX_BIN")
        folder = Path(exe).parent if exe else None
        if not folder:
            raise RuntimeError("Set CPLEX_LIBRARY to the CPLEX shared library or set CPLEX_BIN.")
        files = sorted(p for p in folder.glob("*cplex*") if p.suffix in {".dll", ".so", ".dylib"} and "concert" not in p.name.lower() and p.name.lower().startswith(("cplex","libcplex")))
        if len(files) != 1:
            raise RuntimeError("Set CPLEX_LIBRARY to the installed CPLEX shared library.")
        location = str(files[0])
    lib = ct.CDLL(location)
    def bind(name, result, *args):
        f = getattr(lib, name)
        f.restype = result
        f.argtypes = list(args)
    bind("CPXopenCPLEX", ptr, integers)
    bind("CPXcloseCPLEX", integer, ct.POINTER(ptr))
    bind("CPXcreateprob", ptr,ptr,integers,ct.c_char_p)
    bind("CPXfreeprob",integer,ptr,ct.POINTER(ptr))
    bind("CPXcopylp",integer,ptr,ptr,integer,integer,integer,numbers,numbers,ct.c_char_p,
         integers,integers,integers,numbers,numbers,numbers,numbers)
    bind("CPXsetintparam",integer,ptr,integer,integer)
    bind("CPXsetdblparam",integer,ptr,integer,number)
    bind("CPXlpopt",integer,ptr,ptr)
    bind("CPXgetstat",integer,ptr,ptr)
    bind("CPXsolution",integer,ptr,ptr,integers,numbers,numbers,numbers,numbers,numbers)
    return lib


def check(code):
    if code:
        raise RuntimeError(f"CPLEX returned error {code}")


def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    objective = np.ascontiguousarray(c, dtype=np.float64)
    n = len(c)
    matrices = [sparse.csr_matrix(a) for a in (A_ub,A_eq) if a is not None]
    matrix = sparse.vstack(matrices, format="csc")
    matrix.sum_duplicates()
    rhs = np.concatenate([np.asarray(b,dtype=float) for b in (b_ub,b_eq) if b is not None])
    nu = 0 if b_ub is None else len(b_ub)
    senses = b"L"*nu + b"E"*(len(rhs)-nu)
    starts = np.ascontiguousarray(matrix.indptr[:-1],dtype=np.int32)
    counts = np.ascontiguousarray(np.diff(matrix.indptr),dtype=np.int32)
    indices = np.ascontiguousarray(matrix.indices,dtype=np.int32)
    data = np.ascontiguousarray(matrix.data,dtype=np.float64)
    limits = bounds if bounds is not None else [(0,None)]*n
    lower = np.array([-1e20 if lo is None else lo for lo,hi in limits],dtype=float)
    upper = np.array([1e20 if hi is None else hi for lo,hi in limits],dtype=float)
    lib = library()
    status = integer()
    env = ptr(lib.CPXopenCPLEX(ct.byref(status)))
    check(status.value)
    lp = ptr()
    try:
        lp = ptr(lib.CPXcreateprob(env,ct.byref(status),b"capacity"))
        check(status.value)
        check(lib.CPXsetintparam(env,1067,1))  # Threads
        check(lib.CPXsetdblparam(env,1016,1e-9))  # Feasibility
        check(lib.CPXsetdblparam(env,1014,1e-9))  # Optimality
        check(lib.CPXcopylp(env,lp,n,len(rhs),1,objective.ctypes.data_as(numbers),
              rhs.ctypes.data_as(numbers),senses,starts.ctypes.data_as(integers),
              counts.ctypes.data_as(integers),indices.ctypes.data_as(integers),
              data.ctypes.data_as(numbers),lower.ctypes.data_as(numbers),upper.ctypes.data_as(numbers),None))
        check(lib.CPXlpopt(env,lp))
        code = lib.CPXgetstat(env,lp)
        if code == 3:
            return SimpleNamespace(success=False,status=2,native_status=code,message="infeasible")
        if code != 1:
            raise RuntimeError(f"Unexpected CPLEX solution status {code}")
        x,dual = np.empty(n),np.empty(len(rhs))
        obj = number()
        check(lib.CPXsolution(env,lp,None,ct.byref(obj),x.ctypes.data_as(numbers),
                              dual.ctypes.data_as(numbers),None,None))
        return SimpleNamespace(success=True,status=0,native_status=code,message="optimal",x=x,fun=obj.value,
                               ineqlin=SimpleNamespace(marginals=dual[:nu]))
    finally:
        if lp:
            lib.CPXfreeprob(env,ct.byref(lp))
        lib.CPXcloseCPLEX(ct.byref(env))
