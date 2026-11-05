"""Regression checks for restoring actual mutable AgentDojo runtime state."""
from datetime import datetime
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'provenance'))
from common import BASE, OUTPUT, read, write, sha
from snapshot import restore_environment


def main():
    identity = read(BASE / 'agentdojo-source.json')
    sys.path.insert(0, identity['source'])
    from agentdojo.task_suite.load_suites import get_suites
    from agentdojo.default_suites.v1.tools.slack import send_direct_message
    from agentdojo.default_suites.v1.tools.banking_client import set_balance
    from agentdojo.default_suites.v1.tools.types import CalendarEvent, EvenStatus
    suites = get_suites('v1.2.1')
    checks = []

    def verify(name, environment):
        snapshot = environment.model_dump(mode='json')
        restored = restore_environment(type(environment), snapshot)
        assert restored.model_dump(mode='json') == snapshot
        checks.append(name)
        return restored

    for name, suite in suites.items():
        verify(name + '_default_environment', suite.load_and_inject_default_environment({}))
    for name in ['workspace', 'travel']:
        env = suites[name].load_and_inject_default_environment({})
        event = env.calendar.create_event('Snapshot regression event', 'Runtime-created event',
            datetime(2024, 6, 7, 11), datetime(2024, 6, 7, 12), None, ['snapshot-check@example.com'])
        env.calendar.cancel_event(env.inbox, event.id_)
        new_email = env.inbox.send_email(['snapshot-check@example.com'], 'Snapshot regression email', 'Runtime-created email')
        deleted_email = next(i for i in env.inbox.emails if i != new_email.id_)
        env.inbox.delete_email(deleted_email)
        snapshot = env.model_dump(mode='json')
        # This reproduces the failure that the snapshot loader must prevent.
        assert type(env).model_validate(snapshot).model_dump(mode='json') != snapshot
        restored = verify(name + '_email_deletion_creation_and_calendar_attachment', env)
        assert new_email.id_ in restored.inbox.emails and deleted_email not in restored.inbox.emails
        assert deleted_email in restored.inbox.trash
        assert restored.calendar.events[event.id_].status is EvenStatus.canceled
        attachments = [a for email in restored.inbox.emails.values() for a in email.attachments]
        assert any(isinstance(a, CalendarEvent) and a.id_ == event.id_ for a in attachments)
        if name == 'workspace':
            new_file = env.cloud_drive.create_file('snapshot-regression.txt', 'Runtime-created file')
            deleted_file = next(i for i in env.cloud_drive.files if i != new_file.id_)
            env.cloud_drive.delete_file(deleted_file)
            appended_file = next(i for i in env.cloud_drive.files if i != new_file.id_)
            env.cloud_drive.append_to_file(appended_file, '\nRuntime appended content')
            restored = verify(name + '_file_creation_deletion_and_append', env)
            assert new_file.id_ in restored.cloud_drive.files and deleted_file not in restored.cloud_drive.files
            assert restored.cloud_drive.files[appended_file].content.endswith('\nRuntime appended content')
    env = suites['slack'].load_and_inject_default_environment({})
    recipient = env.slack.users[0]
    send_direct_message(env.slack, recipient, 'Runtime snapshot regression message')
    restored = verify('slack_new_direct_message', env)
    assert restored.slack.user_inbox[recipient][-1].body == 'Runtime snapshot regression message'
    env = suites['banking'].load_and_inject_default_environment({})
    set_balance(env.bank_account, 1234.25)
    restored = verify('banking_changed_balance', env)
    assert restored.bank_account.balance == 1234.25
    receipt = {'status': 'passed', 'checks': checks, 'check_count': len(checks),
        'ordinary_model_validate_state_loss_reproduced': True,
        'all_four_suites_roundtrip_exactly': True, 'created_and_deleted_runtime_objects_preserved': True,
        'calendar_attachment_union_enum_and_datetime_types_preserved': True,
        'snapshot_restore_source_sha256': sha(Path(__file__).with_name('snapshot.py')),
        'verification_source_sha256': sha(Path(__file__)), 'agentdojo_source_sha256': sha(BASE / 'agentdojo-source.json'),
        'verified_at_unix': time.time()}
    write(OUTPUT / 'snapshot_validation.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
