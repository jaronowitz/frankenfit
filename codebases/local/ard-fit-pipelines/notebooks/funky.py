from attrs import define, field


@define(slots=True)
class MyClass:
    """
    Just an example.
    """

    # ... maybe there are a bunch of fields here...

    # then, a "normal" class variable for some other purpose. It seems like I
    # should be able to have a class variable that isn't an attr.field here and
    # there if I want. And in that case, I'd expect it to behave in the usual way,
    # i.e., instances start off with `self.some_state` as a reference to the
    # (global) value in the class, but then, as is often done, they assign some
    # new value to `self.some_state` at which point it becomes an instance
    # variable
    some_state = None

    def do_something(self, *args, **kwargsd):
        # ... do something... then maybe we want to update our internal state:
        if self.some_state is None:
            self.some_state = 0
        else:
            self.some_state += 1


t = MyClass()
print(f"{t=}")
print(f"{t.some_state=}")

# crashes if slots=True
t.do_something()
print(f"{t.some_state=}")
