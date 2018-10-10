Release procedure
==================

- Merge pyup PR updating the dependencies
- Update personal fork & local master to Axfoundation fork
- Edit and commit HISTORY.md
- bumpversion minor
- Push to personal and AxFoundation fork, with --tags
- fast-foward and push AxFoundation/stable
- Add release info on release page of github website