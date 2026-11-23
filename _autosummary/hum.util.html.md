# hum.util

A few general utils

### Functions

| [`getmodulename`](#hum.util.getmodulename)(obj[, default])                    | Get name of module of object                                                           |
|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| [`return_none`](#hum.util.return_none)(\*args, \*\*kwargs)                  | Return None for any arguments.                                                         |
| [`round_number`](#hum.util.round_number)(number[, round_to])                 | Round a number to the nearest multiple of round_to.                                    |
| [`round_numbers`](#hum.util.round_numbers)(items[, round_to, ...])            | Round numbers in an iterable, optionally extracting the number to round from an index. |
| [`scale_frequencies`](#hum.util.scale_frequencies)([scale, tuning, octave_range]) | Get frequencies for a given scale.                                                     |
| [`scale_snapper`](#hum.util.scale_snapper)([scale, tuning, octave_range])     | Snap frequency to the nearest midi note of the midi_nodes list, modulo 12.             |
| [`simple_chunker`](#hum.util.simple_chunker)(a, chk_size, \*[, include_tail])  | Chunks an iterable into non-overlapping chunks of size `chk_size`.                     |
| [`tempered_semitone_frequencies`](#hum.util.tempered_semitone_frequencies)([tuning, ...])     | Get a sorted list of frequencies for a given tuning and octave range.                  |

### Classes

| [`ListSnapper`](#hum.util.ListSnapper)(snap_values)       | Snap to the nearest value in a list.                    |
|---------------------------------------------------------------------------------|---------------------------------------------------------|
| `ModuleNotFoundErrorNiceMessage`([msg])                                         |                                                         |
| `ModuleNotFoundIgnore`()                                                        |                                                         |
| [`SnapToChromatic`](#hum.util.SnapToChromatic)([base_tuning]) | Snap to the nearest chromatic note in a tempered scale. |

### *class* hum.util.ListSnapper(snap_values)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Snap to the nearest value in a list.

```pycon
>>> snapper = ListSnapper([0, 1, 2, 3, 4, 5])
>>> snapper(2.3)
2
>>> snapper(2.7)
3
>>> snapper(2.5)  # exactly the middle of 2 and 3; snaps to 2
2
```

### *class* hum.util.SnapToChromatic(base_tuning=440)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Snap to the nearest chromatic note in a tempered scale.

### hum.util.getmodulename(obj, default='')

Get name of module of object

### hum.util.return_none(\*args, \*\*kwargs)

Return None for any arguments.

### hum.util.round_number(number, round_to=0.001)

Round a number to the nearest multiple of round_to.

### Parameters

- number: float, the number to round.
- round_to: float, the value to round to.

### Returns

- float, the rounded number.

```pycon
>>> round_number(1.234567, 0.01)
1.23
```

Note, the rounder rounds to nearest multiple of round_to, not to the nearest decimal place.

```pycon
>>> round_number(1.234567, 0.001)
1.235
```

### hum.util.round_numbers(items, round_to=0.001, , index_of_item_number=None, egress=None)

Round numbers in an iterable, optionally extracting the number to round from an index.

### Parameters

- items: iterable of numbers or iterable of tuples/lists containing numbers.
- round_to: float, round to the nearest multiple of this value.
- index_of_item_number: int or None. If None, round the item directly;
  otherwise, round the item at this index in each iterable element.

### Returns

- generator yielding items with rounded numbers.

### Examples

```pycon
>>> list(round_numbers([1.23, 3.14159], round_to=0.1))
[1.2, 3.1]
```

```pycon
>>> items = [(1.234, 'one'), (3.14159, 'three')]
>>> list(round_numbers(items, round_to=0.1, index_of_item_number=0))
[[1.2, 'one'], [3.1, 'three']]
```

### hum.util.scale_frequencies(scale=(0, 2, 4, 5, 7, 9, 11), , tuning=440.0, octave_range=(0, 10))

Get frequencies for a given scale.

* **Parameters:**
  * **scale** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`float`](https://docs.python.org/3/builtins/functions.html#float) | [`int`](https://docs.python.org/3/builtins/functions.html#int)]) – A list of integers representing the scale.
    For example, (0, 2, 4, 5, 7, 9, 11) corresponds to the C major scale.
  * **tuning** – The tuning frequency (default is 440 Hz).
  * **octave_range** – A tuple of two integers representing the range of octaves.
* **Returns:**
  A list of frequencies corresponding to the notes in the given scale.

### hum.util.scale_snapper(scale=(0, 2, 4, 5, 7, 9, 11), , tuning=440.0, octave_range=(0, 10))

Snap frequency to the nearest midi note of the midi_nodes list, modulo 12.

### hum.util.simple_chunker(a, chk_size, , include_tail=True)

Chunks an iterable into non-overlapping chunks of size `chk_size`.

#### NOTE
This chunker is simpler, but also less efficient than `chunk_iterable`.
It does have the extra `include_tail` argument, though.
Though note that you can get the effect of `include_tail=False` in `chunk_iterable`
by using `filter(lambda x: len(x) == chk_size, chunk_iterable(...))`.

* **Parameters:**
  * **a** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`T`]) – The iterable to be chunked.
  * **chk_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – The size of each chunk.
  * **include_tail** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, includes the remaining elements as the last chunk
    even if they are fewer than `chk_size`. Defaults to True.
* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`T`, [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]]
* **Returns:**
  An iterator of tuples, where each tuple is a chunk of size `chk_size`
  (or fewer elements if `include_tail` is True).

### Examples

```pycon
>>> list(simple_chunker(range(8), 3))
[(0, 1, 2), (3, 4, 5), (6, 7)]
>>> list(simple_chunker(range(8), 3, include_tail=False))
[(0, 1, 2), (3, 4, 5)]
```

### hum.util.tempered_semitone_frequencies(tuning=440.0, octave_range=(0, 10), digits=5)

Get a sorted list of frequencies for a given tuning and octave range.

* **Parameters:**
  * **tuning** – The tuning frequency (default is 440 Hz).
    It corresponds to A4 (that is, the 69th MIDI note).
  * **octave_range** – 

    A tuple of two integers representing the range of octaves.

    The first integer is the starting octave (inclusive), and the second integer is the ending octave (exclusive).
    For example, (0, 10) will generate frequencies from C0 to B9.
    The default is (0, 10), which covers a wide range of frequencies.
* **Returns:**
  A sorted list of frequencies in the specified octave range. The first (i.e. .[0]) frequency will correspond to a C.

```pycon
>>> frequencies = tempered_semitone_frequencies()
>>> len(frequencies)
120
>>> frequencies
[8.1758, 8.66196, 9.17702, 9.72272, ..., 7040.0, 7458.62018, 7902.13282]
```
