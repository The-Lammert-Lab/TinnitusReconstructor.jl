# Development notes

## Week of 11/21/22

### 3-autoformat-code

- Autoformatting is difficult to do because 
Julia is saved to different locations depending 
on the operating system. As far as I can tell, it can't be 
run directly from the terminal without some user-specific setup.

- `FormatCheck.yml` is responsible for checking the formatting
but does not actually format the code.

### 1-update-cicd-to-test-on-julia-18

- TagBot requires an SSH Deploy Key in order to run
other actions (like documentation). See [here](https://github.com/JuliaRegistries/TagBot#ssh-deploy-keys). 
This is a repository-owner task, I believe.

    - Permissions section in `TagBot.yml` might not be necessary. 
    It's not used in SciML's TagBot! I kept it in because that's what they suggest.

    - I removed the default lookback in workflow_dispatch
    because vscode was giving an error. Couldn't easily sort out
    the error and I don't think we'll be regularly running this action.



