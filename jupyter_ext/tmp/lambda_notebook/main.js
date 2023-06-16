define([
    'base/js/namespace',
    'base/js/events'
], function (Jupyter, events) {
    var is_import_cell_created = false;

    var insert_cell = function () {
        if (is_import_cell_created) {
            Jupyter.notebook.insert_cell_below('code').set_text(`print(inter_obj.process_commands("""\n\n"""))`);
        } else {
            is_import_cell_created = true;
            Jupyter.notebook.insert_cell_below('code').set_text(`from lambda_code_interpreter import LambdaCalculusInterpreter\n\ninter_obj = LambdaCalculusInterpreter()`);
        }
        Jupyter.notebook.execute_cell_and_select_below();
    };
    // Add Toolbar button
    var lambdaNotebookButton = function () {
        console.log();
        Jupyter.toolbar.add_buttons_group([
            Jupyter.keyboard_manager.actions.register({
                'help': 'Add Lambda Notebook cell',
                'icon': 'fa-paper-plane',
                'handler': insert_cell
            }, 'addlambdanotebook-cell', 'Lambda Notebook')
        ])
    }

    var lambdaNotebookKeyboardShiftEnterButton = function () {
        console.log("keyboard pushed")
        Jupyter.keyboard_manager.command_shortcuts.add_shortcut('shift-enter', {
            'help': 'Add Lambda Notebook cell',
            'icon': 'fa-paper-plane',
            'handler': insert_cell
        })
    }

    var lambdaNotebookKeyboardNButton = function () {
        console.log("keyboard pushed")
        Jupyter.keyboard_manager.command_shortcuts.add_shortcut('n', {
            'help': 'Add Lambda Notebook cell',
            'icon': 'fa-paper-plane',
            'handler': insert_cell
        })
    }

    // Run on start
    function load_ipython_extension() {
        // Add a default cell if there are no cells
        if (Jupyter.notebook.get_cells().length === 1) {
            Jupyter.notebook.delete_cell(1);
            insert_cell();
        }
        lambdaNotebookButton();
        lambdaNotebookKeyboardShiftEnterButton();
        lambdaNotebookKeyboardNButton();
    }

    return {
        load_ipython_extension: load_ipython_extension
    };
});