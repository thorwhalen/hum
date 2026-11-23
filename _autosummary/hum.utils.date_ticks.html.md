# hum.utils.date_ticks

Date utils

### Functions

| [`largest_unit_that_changes_at_every_tick`](#hum.utils.date_ticks.largest_unit_that_changes_at_every_tick)(...)   | Returns the largest time unit for which each time tick changes.   |
|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| `str_ticks`(ticks, ticks_unit[, ...])                                                           |                                                                   |
| `strftime_format_for_ticks`(ticks, ticks_unit)                                                  |                                                                   |
| [`strftime_with_precision`](#hum.utils.date_ticks.strftime_with_precision)(tick, format[, ...])   | Returns a formatted string for a datetime (tick).                 |
| `unit_aligned_ticks`(ticks, ticks_unit)                                                         |                                                                   |
| `utc_datetime_from_val_and_unit`(val, unit)                                                     |                                                                   |

### hum.utils.date_ticks.largest_unit_that_changes_at_every_tick(ticks, ticks_unit)

Returns the largest time unit for which each time tick changes.

* **Parameters:**
  * **ticks** – The list of ticks
  * **ticks_unit** – 

    The unit of the elements of ticks, expressed in seconds. For example, if the list
    contains hours, unit=3600, if minutes, unit=60, if seconds unit=1,
    if milliseconds unit=0.001.
    > Note:
    > : You can also use a string to express the unit, as long as it’s recognized by the
    >   unit_str_to_unit_in_seconds dict. Keys recognized:
    >   [‘day’, ‘hour’, ‘h’, ‘minute’, ‘mn’, ‘second’, ‘s’, ‘millisecond’, ‘ms’, ‘microsecond’, ‘us’]
* **Returns:**

### hum.utils.date_ticks.strftime_with_precision(tick, format, sub_secs_precision=2)

Returns a formatted string for a datetime (tick).

* **Parameters:**
  * **tick** – The datetime for this tick
  * **format** – The formatting string
  * **sub_secs_precision** – Number of digits to used for sub-seconds.
    If None, will choose it “smartly/dynamically”
* **Returns:**
  Formatted string
