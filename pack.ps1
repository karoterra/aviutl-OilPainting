$gitTag = (git tag --points-at)
$archiveName = "aviutl-OilPainting"
if (![string]::IsNullOrEmpty($gitTag))
{
    $archiveName = "${archiveName}_${gitTag}"
}

New-Item publish -ItemType Directory -Force

7z a "publish\${archiveName}.zip" `
    .\README.md `
    .\CHANGELOG.md `
    .\LICENSE `
    .\CREDITS.md `
    ".\script\油絵.anm" `
    ".\build\Release\KaroterraOilPainting.dll"
