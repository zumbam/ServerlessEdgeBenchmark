# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.12
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

"""
This documentation was automatically generated using original comments in
Doxygen format. As some C types and data structures cannot be directly mapped
into Python types, some non-trivial type conversion could have place.
Basically a type is replaced with another one that has the closest match, and
sometimes one argument of generated function comprises several arguments of the
original function (usually two).

Functions having error code as the return value and returning effective
value in one of its arguments are transformed so that the effective value is
returned in a regular fashion and run-time exception is being thrown in case of
negative error code.
"""


from sys import version_info as _swig_python_version_info
if _swig_python_version_info >= (2, 7, 0):
    def swig_import_helper():
        import importlib
        pkg = __name__.rpartition('.')[0]
        mname = '.'.join((pkg, '_ad_alsa')).lstrip('.')
        try:
            return importlib.import_module(mname)
        except ImportError:
            return importlib.import_module('_ad_alsa')
    _ad_alsa = swig_import_helper()
    del swig_import_helper
elif _swig_python_version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_ad_alsa', [dirname(__file__)])
        except ImportError:
            import _ad_alsa
            return _ad_alsa
        try:
            _mod = imp.load_module('_ad_alsa', fp, pathname, description)
        finally:
            if fp is not None:
                fp.close()
        return _mod
    _ad_alsa = swig_import_helper()
    del swig_import_helper
else:
    import _ad_alsa
del _swig_python_version_info

try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        object.__setattr__(self, name, value)
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr(self, class_type, name):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    raise AttributeError("'%s' object has no attribute '%s'" % (class_type.__name__, name))


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_method(set):
    def set_attr(self, name, value):
        if (name == "thisown"):
            return self.this.own(value)
        if hasattr(self, name) or (name == "this"):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add attributes to %s" % self)
    return set_attr


class Ad(object):
    """Proxy of C Ad struct."""

    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, audio_device=None, sampling_rate=16000):
        """__init__(Ad self, char const * audio_device=None, int sampling_rate=16000) -> Ad"""
        this = _ad_alsa.new_Ad(audio_device, sampling_rate)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this
    __swig_destroy__ = _ad_alsa.delete_Ad
    __del__ = lambda self: None

    def __enter__(self):
        """__enter__(Ad self) -> Ad"""
        return _ad_alsa.Ad___enter__(self)


    def __exit__(self, exception_type, exception_value, exception_traceback):
        """__exit__(Ad self, PyObject * exception_type, PyObject * exception_value, PyObject * exception_traceback)"""
        return _ad_alsa.Ad___exit__(self, exception_type, exception_value, exception_traceback)


    def start_recording(self):
        """start_recording(Ad self) -> int"""
        return _ad_alsa.Ad_start_recording(self)


    def stop_recording(self):
        """stop_recording(Ad self) -> int"""
        return _ad_alsa.Ad_stop_recording(self)


    def readinto(self, DATA):
        """readinto(Ad self, char * DATA) -> int"""
        return _ad_alsa.Ad_readinto(self, DATA)

Ad_swigregister = _ad_alsa.Ad_swigregister
Ad_swigregister(Ad)



