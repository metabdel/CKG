import os
from apps import basicApp
from report_manager import project
import dash_html_components as html
import dash_core_components as dcc


class ProjectApp(basicApp.BasicApp):
    """
    Defines what a project App is in the report_manager.
    Includes multiple tabs for different data types.
    """
    def __init__(self, projectId, title, subtitle, description, layout = [], logo = None, footer = None, force=False):
        self._project_id = projectId
        self._page_type = "projectPage"
        self._force = force
        basicApp.BasicApp.__init__(self, title, subtitle, description, self.page_type, layout, logo, footer)
        self.build_page()

    @property
    def project_id(self):
        """
        Retrieves project identifier.
        """
        return self._project_id

    @project_id.setter
    def project_id(self, project_id):
        """
        Sets 'project_id' input value as project_id property of the class.

        :param str project_id: project identifier.
        """
        self._project_id = project_id
        
    @property
    def force(self):
        """
        Retrieves attribute force (whether or not the project report needs to be regenerated).
        """
        return self._force

    @force.setter
    def force(self, force):
        """
        Sets 'force' value as force property of the class.

        :param boolean force: force.
        """
        self._force = force
        
    def build_header(self):
        buttons = html.Div([html.Div([html.A('Download Project Report',
                                id='download-zip',
                                href="",
                                target="_blank",
                                n_clicks = 0,
                                className="button_link"
                                )]),
                            html.Div([html.A("Regenerate Project Report", 
                                id='regenerate', 
                                href='', 
                                target='', 
                                n_clicks=0,
                                className="button_link")])])
        
        return buttons

    def build_page(self):
        """
        Builds project and generates the report.
        For each data type in the report (e.g. 'proteomics', 'clinical'), \
        creates a designated tab.
        A button to download the entire project and report is added.
        """
        p = project.Project(self.project_id, datasets={}, knowledge=None, report={})
        p.build_project(self.force)
        p.generate_report()
        
        if p.name is not None:
            self.title = "Project: {}".format(p.name)
        else:
            self.title = ''
        self.add_basic_layout()
        
        plots = p.show_report("app")
        
        tabs = []
        buttons = self.build_header()
        
        self.add_to_layout(buttons)
        for data_type in plots:
            if len(plots[data_type]) >=1:
                tab_content = [html.Div(plots[data_type])]
                tab = dcc.Tab(tab_content, label=data_type)
                tabs.append(tab)
        lc = dcc.Tabs(tabs)
        self.add_to_layout(lc)
