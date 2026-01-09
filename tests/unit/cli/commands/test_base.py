"""Tests for CLI command base infrastructure."""

import pytest

from statgpt.cli.commands.base import Command, CommandArg, CommandGroup, CommandRegistry


class TestCommandArgParsing:
    """Tests for Command.parse_args() method."""

    def test_empty_input_returns_defaults(self, sample_command):
        """Empty input should return default values for non-required args."""
        # sample_command has required 'name' arg, so we need a command without required args
        cmd = Command(
            name="test",
            description="Test",
            handler=sample_command.handler,
            args=[
                CommandArg(name="mode", description="Mode", default="default"),
                CommandArg(name="flag", description="Flag", is_flag=True),
            ],
        )
        result = cmd.parse_args("")
        assert result == {"mode": "default", "flag": False}

    def test_long_arg_with_value(self, simple_command):
        """Parse long argument with value."""
        cmd = Command(
            name="test",
            description="Test",
            handler=simple_command.handler,
            args=[CommandArg(name="name", description="Name")],
        )
        result = cmd.parse_args("--name value")
        assert result == {"name": "value"}

    def test_short_arg_with_value(self, simple_command):
        """Parse short argument with value."""
        cmd = Command(
            name="test",
            description="Test",
            handler=simple_command.handler,
            args=[CommandArg(name="name", description="Name", short_name="n")],
        )
        result = cmd.parse_args("-n value")
        assert result == {"name": "value"}

    def test_flag_argument(self, simple_command):
        """Parse flag argument."""
        cmd = Command(
            name="test",
            description="Test",
            handler=simple_command.handler,
            args=[CommandArg(name="clean", description="Clean", is_flag=True)],
        )
        result = cmd.parse_args("--clean")
        assert result == {"clean": True}

    def test_flag_argument_default_false(self, simple_command):
        """Flag argument defaults to False when not provided."""
        cmd = Command(
            name="test",
            description="Test",
            handler=simple_command.handler,
            args=[CommandArg(name="clean", description="Clean", is_flag=True)],
        )
        result = cmd.parse_args("")
        assert result == {"clean": False}

    def test_multiple_arguments(self, simple_command):
        """Parse multiple arguments together."""
        cmd = Command(
            name="test",
            description="Test",
            handler=simple_command.handler,
            args=[
                CommandArg(name="name", description="Name"),
                CommandArg(name="clean", description="Clean", is_flag=True),
            ],
        )
        result = cmd.parse_args("--name value --clean")
        assert result == {"name": "value", "clean": True}

    def test_missing_required_arg_raises_error(self, sample_command):
        """Missing required argument should raise ValueError."""
        with pytest.raises(ValueError, match="Missing required argument: --name"):
            sample_command.parse_args("")

    def test_invalid_choice_raises_error(self, simple_command):
        """Invalid choice value should raise ValueError."""
        cmd = Command(
            name="test",
            description="Test",
            handler=simple_command.handler,
            args=[CommandArg(name="mode", description="Mode", choices=["a", "b"])],
        )
        with pytest.raises(ValueError, match="Invalid value for --mode: invalid"):
            cmd.parse_args("--mode invalid")

    def test_invalid_choice_short_arg_raises_error(self, simple_command):
        """Invalid choice value with short arg should raise ValueError."""
        cmd = Command(
            name="test",
            description="Test",
            handler=simple_command.handler,
            args=[CommandArg(name="mode", description="Mode", short_name="m", choices=["a", "b"])],
        )
        with pytest.raises(ValueError, match="Invalid value for -m: invalid"):
            cmd.parse_args("-m invalid")

    def test_unknown_argument_raises_error(self, simple_command):
        """Unknown argument should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown argument: --unknown"):
            simple_command.parse_args("--unknown value")

    def test_unknown_short_argument_raises_error(self, simple_command):
        """Unknown short argument should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown argument: -x"):
            simple_command.parse_args("-x value")

    def test_quoted_string_value(self, simple_command):
        """Parse quoted string value."""
        cmd = Command(
            name="test",
            description="Test",
            handler=simple_command.handler,
            args=[CommandArg(name="name", description="Name")],
        )
        result = cmd.parse_args('--name "hello world"')
        assert result == {"name": "hello world"}

    def test_arg_without_value_raises_error(self, simple_command):
        """Non-flag argument without value should raise ValueError."""
        cmd = Command(
            name="test",
            description="Test",
            handler=simple_command.handler,
            args=[CommandArg(name="name", description="Name")],
        )
        with pytest.raises(ValueError, match="Argument --name requires a value"):
            cmd.parse_args("--name")

    def test_short_arg_without_value_raises_error(self, simple_command):
        """Short non-flag argument without value should raise ValueError."""
        cmd = Command(
            name="test",
            description="Test",
            handler=simple_command.handler,
            args=[CommandArg(name="name", description="Name", short_name="n")],
        )
        with pytest.raises(ValueError, match="Argument -n requires a value"):
            cmd.parse_args("-n")

    def test_hyphenated_arg_name(self, simple_command):
        """Hyphenated argument name should be converted to underscores."""
        cmd = Command(
            name="test",
            description="Test",
            handler=simple_command.handler,
            args=[CommandArg(name="data-source", description="Data source")],
        )
        result = cmd.parse_args("--data-source value")
        assert result == {"data_source": "value"}

    def test_unexpected_token_raises_error(self, simple_command):
        """Unexpected token (not starting with -) should raise ValueError."""
        with pytest.raises(ValueError, match="Unexpected token: value"):
            simple_command.parse_args("value")

    def test_invalid_syntax_raises_error(self, simple_command):
        """Invalid shlex syntax should raise ValueError."""
        cmd = Command(
            name="test",
            description="Test",
            handler=simple_command.handler,
            args=[CommandArg(name="name", description="Name")],
        )
        with pytest.raises(ValueError, match="Invalid argument syntax"):
            cmd.parse_args('--name "unclosed')

    def test_valid_choice(self, simple_command):
        """Valid choice should be accepted."""
        cmd = Command(
            name="test",
            description="Test",
            handler=simple_command.handler,
            args=[CommandArg(name="mode", description="Mode", choices=["a", "b"])],
        )
        result = cmd.parse_args("--mode a")
        assert result == {"mode": "a"}


class TestCommandGetHelp:
    """Tests for Command.get_help() method."""

    def test_help_contains_name_and_description(self, simple_command):
        """Help text should contain command name and description."""
        help_text = simple_command.get_help()
        assert "simple" in help_text
        assert "Simple command" in help_text

    def test_help_with_required_arg(self, noop_handler):
        """Help text should mark required arguments."""
        cmd = Command(
            name="test",
            description="Test",
            handler=noop_handler,
            args=[CommandArg(name="name", description="Name", required=True)],
        )
        help_text = cmd.get_help()
        assert "(required)" in help_text

    def test_help_with_default_value(self, noop_handler):
        """Help text should show default values."""
        cmd = Command(
            name="test",
            description="Test",
            handler=noop_handler,
            args=[CommandArg(name="mode", description="Mode", default="default")],
        )
        help_text = cmd.get_help()
        assert "(default: default)" in help_text

    def test_help_with_choices(self, noop_handler):
        """Help text should show available choices."""
        cmd = Command(
            name="test",
            description="Test",
            handler=noop_handler,
            args=[CommandArg(name="mode", description="Mode", choices=["a", "b"])],
        )
        help_text = cmd.get_help()
        assert "[a/b]" in help_text

    def test_help_with_short_name(self, noop_handler):
        """Help text should show short name."""
        cmd = Command(
            name="test",
            description="Test",
            handler=noop_handler,
            args=[CommandArg(name="name", description="Name", short_name="n")],
        )
        help_text = cmd.get_help()
        assert "-n" in help_text
        assert "--name" in help_text


class TestCommandFullName:
    """Tests for Command.full_name property."""

    def test_standalone_command_full_name(self, simple_command):
        """Standalone command full name is just the name."""
        assert simple_command.full_name == "simple"

    def test_group_command_full_name(self, noop_handler):
        """Group command full name includes group name."""
        cmd = Command(name="sub", description="Sub", handler=noop_handler, group="parent")
        assert cmd.full_name == "parent sub"


class TestCommandGroup:
    """Tests for CommandGroup class."""

    def test_add_command_sets_group(self, noop_handler):
        """Adding command to group should set its group property."""
        group = CommandGroup(name="test", description="Test")
        cmd = Command(name="sub", description="Sub", handler=noop_handler)
        group.add_command(cmd)
        assert cmd.group == "test"
        assert "sub" in group.commands

    def test_get_help(self, sample_group):
        """Get help should include group name and all subcommands."""
        help_text = sample_group.get_help()
        assert "test" in help_text
        assert "sub1" in help_text
        assert "sub2" in help_text


class TestCommandRegistry:
    """Tests for CommandRegistry class."""

    def test_register_command(self, simple_command):
        """Registered command should be accessible."""
        registry = CommandRegistry()
        registry.register_command(simple_command)
        assert registry.get_command("simple") is simple_command

    def test_register_group(self, sample_group):
        """Registered group and its commands should be accessible."""
        registry = CommandRegistry()
        registry.register_group(sample_group)
        assert registry.get_group("test") is sample_group
        assert registry.get_command("test sub1") is not None

    def test_get_command_standalone(self, registry_with_commands):
        """Get standalone command by name."""
        cmd = registry_with_commands.get_command("simple")
        assert cmd is not None
        assert cmd.name == "simple"

    def test_get_command_group_subcommand(self, registry_with_commands):
        """Get group subcommand by full name."""
        cmd = registry_with_commands.get_command("test sub1")
        assert cmd is not None
        assert cmd.name == "sub1"

    def test_get_command_unknown(self, registry_with_commands):
        """Unknown command should return None."""
        assert registry_with_commands.get_command("unknown") is None

    def test_get_command_unknown_subcommand(self, registry_with_commands):
        """Unknown subcommand should return None."""
        assert registry_with_commands.get_command("test unknown") is None

    def test_get_all_command_names(self, registry_with_commands):
        """Get all command names should return sorted list."""
        names = registry_with_commands.get_all_command_names()
        assert "simple" in names
        assert "test" in names
        assert "test sub1" in names
        assert "test sub2" in names
        assert names == sorted(names)

    def test_completions_empty_input(self, registry_with_commands):
        """Empty input should return all top-level commands and groups."""
        completions = registry_with_commands.get_completions("")
        names = [c[0] for c in completions]
        assert "simple" in names
        assert "test" in names

    def test_completions_partial_match(self, registry_with_commands):
        """Partial input should return matching commands."""
        completions = registry_with_commands.get_completions("si")
        names = [c[0] for c in completions]
        assert "simple" in names
        assert "test" not in names

    def test_completions_group_with_space(self, registry_with_commands):
        """Group name followed by space should return subcommands."""
        completions = registry_with_commands.get_completions("test ")
        names = [c[0] for c in completions]
        assert "sub1" in names
        assert "sub2" in names

    def test_completions_partial_subcommand(self, registry_with_commands):
        """Partial subcommand should return matching subcommands."""
        completions = registry_with_commands.get_completions("test sub1")
        names = [c[0] for c in completions]
        assert "test sub1" in names

    def test_get_help(self, registry_with_commands):
        """Get help should include all commands and groups."""
        help_text = registry_with_commands.get_help()
        assert "simple" in help_text
        assert "test" in help_text
        assert "sub1" in help_text
        assert "sub2" in help_text

    def test_completions_case_insensitive(self, registry_with_commands):
        """Completions should be case insensitive."""
        completions = registry_with_commands.get_completions("SI")
        names = [c[0] for c in completions]
        assert "simple" in names


class TestCommandArgDataclass:
    """Tests for CommandArg dataclass."""

    def test_default_values(self):
        """CommandArg should have sensible defaults."""
        arg = CommandArg(name="test", description="Test arg")
        assert arg.short_name is None
        assert arg.required is False
        assert arg.default is None
        assert arg.choices is None
        assert arg.is_flag is False

    def test_all_values(self):
        """CommandArg should accept all values."""
        arg = CommandArg(
            name="test",
            description="Test arg",
            short_name="t",
            required=True,
            default="default",
            choices=["a", "b"],
            is_flag=False,
        )
        assert arg.name == "test"
        assert arg.description == "Test arg"
        assert arg.short_name == "t"
        assert arg.required is True
        assert arg.default == "default"
        assert arg.choices == ["a", "b"]
        assert arg.is_flag is False
