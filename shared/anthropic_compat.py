"""Compatibility shim for the installed Anthropic SDK.

This environment's `anthropic` SDK (1.3.0) removed the sampling params
(`temperature`, `top_p`, `top_k`) from `messages.create` - newer Claude models
don't accept them. Some frameworks (AutoGen, CrewAI's native path) hardcode
`temperature=1.0` into every request and give no way to turn it off, so they
crash with `unexpected keyword argument 'temperature'`.

Importing this module patches `Messages.create` / `AsyncMessages.create` to drop
those keys before the call. In a normal environment none of this is needed - it
only bridges the newer SDK. Call install() once, early.
"""

_UNSUPPORTED = ("temperature", "top_p", "top_k")
_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    from anthropic.resources.messages import AsyncMessages, Messages

    def _strip(fn):
        def wrapper(self, *args, **kwargs):
            for key in _UNSUPPORTED:
                kwargs.pop(key, None)
            return fn(self, *args, **kwargs)

        return wrapper

    Messages.create = _strip(Messages.create)
    AsyncMessages.create = _strip(AsyncMessages.create)
    _installed = True
