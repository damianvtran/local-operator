"""Chart messages needed by app handlers even before the optional view loads.

Keep this leaf free of renderer/registry imports: runtime annotations and
Textual routing need a real class without pulling the chart into normal boot.
"""

from textual.message import Message


class OrgChartViewDismissed(Message):
    """The page's ``esc`` hint was clicked. The app owns leaving the mode.

    A DEDICATED message (not the subagent view's) so the app's Esc-chain routes
    the click to :meth:`_close_org_chart_view` — reusing the subagent message
    would hit that mode's handler, which does not own this widget.
    """
