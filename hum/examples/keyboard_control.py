#!/usr/bin/env python3
"""
Keyboard event reader that allows detecting keypress events.
Run this script directly in a terminal.
"""

import time
import json
import os
import importlib
from typing import Dict, Any, Optional, Generator, Union, Callable
import argparse


def _load_callback_from_string(callback_path: str) -> Callable:
    """
    Load a callback function from a module path string.

    Format: "module.path:function_name"

    Args:
        callback_path: String path to the callback function

    Returns:
        The callback function

    Raises:
        ValueError: If the path format is invalid
        ImportError: If the module or function cannot be imported
    """
    if ":" not in callback_path:
        raise ValueError("Callback path must be in format 'module.path:function_name'")

    module_path, function_name = callback_path.split(":", 1)

    try:
        module = importlib.import_module(module_path)
        callback_func = getattr(module, function_name)

        if not callable(callback_func):
            raise ValueError(f"Object {function_name} in {module_path} is not callable")

        return callback_func
    except ImportError as e:
        raise ImportError(f"Could not import module {module_path}: {e}")
    except AttributeError as e:
        raise ImportError(
            f"Could not find function {function_name} in module {module_path}: {e}"
        )


def _load_arg_mapping(arg_mapping: Union[str, Dict]) -> Dict:
    """
    Load argument mapping from a string (file path or JSON string) or return the dict directly.

    Args:
        arg_mapping: String path to JSON file, JSON string, or dict

    Returns:
        Dict of argument mappings
    """
    if isinstance(arg_mapping, dict):
        return arg_mapping

    if isinstance(arg_mapping, str):
        # If it's a file path
        if os.path.isfile(arg_mapping):
            with open(arg_mapping, "r") as f:
                return json.load(f)
        # Otherwise treat as JSON string
        try:
            return json.loads(arg_mapping)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")

    raise TypeError("arg_mapping must be a dict, file path, or JSON string")


def _initialize_pyo_synth(synth_func):
    """
    Initialize a pyo synth function with a server if it's a Synth object.

    Returns the initialized synth function that's ready to play sounds.
    """
    # Try to detect if this is a Synth instance from hum.pyo_util
    if hasattr(synth_func, "start") and hasattr(synth_func, "stop"):
        # This is likely a Synth instance, start it
        synth_func.start()
        return synth_func

    # For raw pyo synth functions, we need to create a synth instance
    # Try to import Synth from hum.pyo_util
    try:
        from hum.pyo_util import Synth

        synth_instance = Synth(synth_func)
        synth_instance.start()
        return synth_instance
    except ImportError:
        # If we can't import Synth, just return the original function
        return synth_func


def keyboard_reader(
    read_rate: float = 0.1,
    exit_key: str = "escape",
    callback: Optional[Union[Callable, str]] = None,
    arg_mapping: Optional[Union[Dict, str]] = None,
    debug: bool = False,
) -> Generator[Optional[Dict[str, Any]], None, None]:
    """
    Generator that yields keyboard events at specified intervals.

    Args:
        read_rate: Time in seconds between checks. Defaults to 0.1.
        exit_key: Key that will trigger exit when using the default example. Defaults to "escape".
        callback: Optional callback function to call when keys are pressed. Can be a callable
            or a string path to a callable in the format "module.path:function_name".
        arg_mapping: Optional mapping from raw keys to callback arguments. Can be a dict,
            a path to a JSON file, or a JSON string.
        debug: Enable debug output for troubleshooting sound issues.

    Yields:
        Dict with key event information or None if no key was pressed

    Example:
        >>> # Usage example (not a doctest)
        >>> reader = keyboard_reader()
        >>> for event in reader:
        ...     if event:
        ...         print(f"Key pressed: {event['key']}")
        ...     if event and event['key'] == 'escape':
        ...         break
    """
    try:
        from pynput import keyboard
    except ImportError:
        print("This script requires the pynput library.")
        print("Install it with: pip install pynput")
        raise

    # Process callback if provided
    callback_func = None
    synth_instance = None
    if callback is not None:
        if isinstance(callback, str):
            callback_func = _load_callback_from_string(callback)
        elif callable(callback):
            callback_func = callback
        else:
            raise TypeError(
                "Callback must be a callable or a string path to a callable"
            )

        # Initialize the synth if it's a pyo synth
        if "pyo_synths" in str(callback_func) or hasattr(callback_func, "start"):
            if debug:
                print(f"Initializing synth function: {callback_func}")
            synth_instance = _initialize_pyo_synth(callback_func)
            callback_func = synth_instance

    # Process arg_mapping if provided
    mapping = None
    if arg_mapping is not None:
        mapping = _load_arg_mapping(arg_mapping)
        if debug:
            print(f"Loaded mapping: {mapping}")

    # Event queue to store key presses
    key_events = []

    def on_press(key):
        """Callback for key press events"""
        try:
            # Regular key presses (alphanumeric keys)
            key_char = key.char
        except AttributeError:
            # Special keys (arrows, esc, etc.)
            # Create a mapping for special keys that need specific names
            special_keys = {
                "Key.esc": "escape",
                "Key.space": "space",
                "Key.enter": "enter",
                "Key.tab": "tab",
            }

            key_str = str(key)
            if key_str in special_keys:
                key_char = special_keys[key_str]
            else:
                key_char = key_str.replace("Key.", "")

        key_info = {"key": key_char, "timestamp": time.time(), "raw_key": str(key)}
        key_events.append(key_info)

        # Call the callback function if provided
        if callback_func:
            try:
                if mapping is not None:
                    # Look up the key in the mapping
                    if key_char in mapping:
                        if debug:
                            print(
                                f"Key {key_char} found in mapping: {mapping[key_char]}"
                            )

                        # If it's a simple value, convert to dict with 'freq' key
                        if isinstance(mapping[key_char], (int, float)):
                            if debug:
                                print(f"Playing freq={mapping[key_char]}")
                            if hasattr(callback_func, "update"):
                                # This is likely a Synth instance
                                callback_func.update({"freq": mapping[key_char]})
                            else:
                                callback_func(freq=mapping[key_char])
                        # If it's already a dict, pass it directly
                        elif isinstance(mapping[key_char], dict):
                            if debug:
                                print(f"Playing with params: {mapping[key_char]}")
                            if hasattr(callback_func, "update"):
                                callback_func.update(mapping[key_char])
                            else:
                                callback_func(**mapping[key_char])
                        # Otherwise just pass the value
                        else:
                            if debug:
                                print(f"Playing with single value: {mapping[key_char]}")
                            callback_func(mapping[key_char])
                    elif debug:
                        print(f"Key {key_char} not found in mapping")
                else:
                    # Just pass the raw key if no mapping
                    if debug:
                        print(
                            f"No mapping, calling with raw key: {key_info['raw_key']}"
                        )
                    callback_func(key_info["raw_key"])
            except Exception as e:
                print(f"Error in callback function: {e}")

    # Start the listener in a non-blocking way
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("Keyboard listener started. Press keys to see events.")
    print(f"Press '{exit_key}' to quit the example.")

    if synth_instance:
        print("Synth initialized and ready to play sounds.")
        print("Make sure your audio is turned on and at a reasonable volume.")

    try:
        while True:
            if key_events:
                yield key_events.pop(0)
            else:
                yield None

            time.sleep(read_rate)
    except KeyboardInterrupt:
        print("\nKeyboard listener stopped.")
    finally:
        # Ensure we stop the listener when we're done
        listener.stop()

        # Clean up synth instance if we created one
        if synth_instance and hasattr(synth_instance, "stop"):
            synth_instance.stop()


def main():
    """Run a simple demo of the keyboard reader"""
    parser = argparse.ArgumentParser(description="Monitor keyboard events")
    parser.add_argument(
        "-r",
        "--rate",
        type=float,
        default=0.1,
        help="Poll rate in seconds (default: 0.1)",
    )
    parser.add_argument(
        "-e",
        "--exit-key",
        type=str,
        default="escape",
        help="Key that will exit the program (default: escape)",
    )
    parser.add_argument(
        "-c",
        "--callback",
        type=str,
        help="Callback function in format 'module.path:function_name'",
    )
    parser.add_argument(
        "-m",
        "--arg-mapping",
        type=str,
        help="Mapping from keys to callback arguments (JSON file path or JSON string)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    args = parser.parse_args()

    # Create the reader with the specified parameters
    reader = keyboard_reader(
        read_rate=args.rate,
        exit_key=args.exit_key,
        callback=args.callback,
        arg_mapping=args.arg_mapping,
        debug=args.debug,
    )

    print(f"Polling for keyboard events every {args.rate} seconds.")
    print(f"Press '{args.exit_key}' to quit.")

    try:
        for event in reader:
            if event:
                print(f"Event: {event}")

                # Exit on specified exit key press
                if event["key"] == args.exit_key:
                    print("Quitting...")
                    break
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
